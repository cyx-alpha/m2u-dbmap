import os
import json
import argparse
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from utils import *
from loader import EEGDataLoader
from models.model_M2M import MainModel


class OneFoldTrainer:
    def __init__(self, args, fold, config):
        self.args = args
        self.fold = fold
        
        self.cfg = config
        self.ds_cfg = config['dataset']
        self.tp_cfg = config['training_params']
        self.es_cfg = self.tp_cfg['early_stopping']
        
        # 获取模态列表
        self.modalities = self.cfg['modalities']
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('[INFO] Config name: {}'.format(config['name']))

        self.train_iter = 0
        self.model = self.build_model()
        self.loader_dict = self.build_dataloader()
        
        self.criterion = nn.CrossEntropyLoss()
        self.activate_train_mode()
        self.optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=self.tp_cfg['lr'], weight_decay=self.tp_cfg['weight_decay'])
        
        self.ckpt_path = os.path.join('checkpoints', config['name'])
        self.ckpt_name = 'ckpt_fold-{0:02d}.pth'.format(self.fold)
        self.early_stopping = EarlyStopping(patience=self.es_cfg['patience'], verbose=True, ckpt_path=self.ckpt_path, ckpt_name=self.ckpt_name, mode=self.es_cfg['mode'])
        

    def build_model(self):
        model = MainModel(self.cfg)
        print('[INFO] Number of params of model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
        model = torch.nn.DataParallel(model, device_ids=list(range(len(self.args.gpu.split(",")))))
        
        if self.tp_cfg['mode'] != 'scratch':
            print('[INFO] Model loaded for multimodal finetune')
            self.load_pretrain_weights(model)

        model.to(self.device)
        print('[INFO] Model prepared, Device used: {} GPU:{}'.format(self.device, self.args.gpu))
        return model

    def load_pretrain_weights(self, model):
        """根据配置中的模态列表，为每个模态加载预训练权重"""
        # 直接构建预训练权重的根目录
        pretrain_root_path = os.path.join('checkpoints', self.ds_cfg['name'], 'single_pretrain')

        for m in self.modalities:
            # 从 channel_mapping 获取真实的通道名
            channel_name = self.ds_cfg['channel_mapping'][m]
            channel_folder = channel_name + '_pretrain'
            
            # 动态构建预训练权重路径
            load_path = os.path.join(pretrain_root_path, channel_folder, 'ckpt_fold-{0:02d}.pth'.format(self.fold))
            print(f'[INFO] Loading pretrain weights for modality "{m}" from: {load_path}')

            if os.path.exists(load_path):
                state_dict = torch.load(load_path, map_location=self.device)
                filtered_dict = {}
                for k, v in state_dict.items():
                    if 'feature' in k:
                        # 动态替换键名以匹配M2M模型中的模块名
                        new_key_single = k.replace('module.feature', f'module.feature_single.{m}')
                        filtered_dict[new_key_single] = v
                        new_key_multi = k.replace('module.feature', f'module.feature_multi.{m}')
                        filtered_dict[new_key_multi] = v
                
                model.load_state_dict(filtered_dict, strict=True)
                print(f'[INFO] Pretrained weights for "{m}" loaded to both single and multi networks successfully')
            else:
                print(f'[WARNING] Pretrained weights for modality "{m}" not found at {load_path}')
        
        print('[INFO] Multi networks initialized with single-modal pretrained weights')
        
    def build_dataloader(self):
        # 假设 EEGDataLoader 已经更新以处理新的配置格式
        train_dataset = EEGDataLoader(self.cfg, self.fold, set_name='train')
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.tp_cfg['batch_size'], shuffle=True, num_workers=4*len(self.args.gpu.split(",")), pin_memory=True)
        val_dataset = EEGDataLoader(self.cfg, self.fold, set_name='val')
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.tp_cfg['batch_size'], shuffle=False, num_workers=4*len(self.args.gpu.split(",")), pin_memory=True)
        test_dataset = EEGDataLoader(self.cfg, self.fold, set_name='test')
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.tp_cfg['batch_size'], shuffle=False, num_workers=4*len(self.args.gpu.split(",")), pin_memory=True)
        print('[INFO] Dataloader prepared')

        return {'train': train_loader, 'val': val_loader, 'test': test_loader}
    
    def activate_train_mode(self):
        self.model.train()

    def _prepare_inputs(self, inputs):
        """动态准备所有模态的输入数据"""
        processed_inputs = {}
        for m in self.modalities:
            # 模型只使用时域信号
            processed_inputs[f'{m}_time'] = inputs[f'{m}_time'].to(self.device)
        return processed_inputs

    def train_one_epoch(self, epoch):
        correct, total, train_loss = 0, 0, 0

        for i, (inputs, labels) in enumerate(self.loader_dict['train']):
            total += labels.size(0)
            
            processed_inputs = self._prepare_inputs(inputs)
            labels = labels.view(-1).to(self.device)
            
            outputs, loss = self.model(processed_inputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            predicted = torch.argmax(outputs, 1)
            correct += predicted.eq(labels).sum().item()
            self.train_iter += 1

            progress_bar(i, len(self.loader_dict['train']), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss / (i + 1), 100. * correct / total, correct, total))
            
            if self.train_iter % self.tp_cfg['val_period'] == 0:
                print('')
                val_acc, val_loss = self.evaluate(mode='val')
                self.early_stopping(val_acc, val_loss, self.model)
                self.activate_train_mode()
                if self.early_stopping.early_stop:
                    break
            
    @torch.no_grad()
    def evaluate(self, mode):
        self.model.eval()
        correct, total, eval_loss = 0, 0, 0
        y_true = np.zeros(0)
        y_pred = np.zeros((0, self.cfg['classifier']['num_classes']))

        for i, (inputs, labels) in enumerate(self.loader_dict[mode]):
            total += labels.size(0)
            
            processed_inputs = self._prepare_inputs(inputs)
            labels = labels.view(-1).to(self.device)

            outputs, loss = self.model(processed_inputs, labels)

            eval_loss += loss.item()
            predicted = torch.argmax(outputs, 1)
            correct += predicted.eq(labels).sum().item()
            y_true = np.concatenate([y_true, labels.cpu().numpy()])
            y_pred = np.concatenate([y_pred, outputs.cpu().numpy()])

            progress_bar(i, len(self.loader_dict[mode]), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (eval_loss / (i + 1), 100. * correct / total, correct, total))

        if mode == 'val':
            return 100. * correct / total, eval_loss
        elif mode == 'test':
            return y_true, y_pred
        else:
            raise NotImplementedError
    
    def run(self):
        for epoch in range(self.tp_cfg['max_epochs']):
            print('\n[INFO] Fold: {}, Epoch: {}'.format(self.fold, epoch))
            self.train_one_epoch(epoch)
            if self.early_stopping.early_stop:
                break
        
        self.model.load_state_dict(torch.load(os.path.join(self.ckpt_path, self.ckpt_name)))
        y_true, y_pred = self.evaluate(mode='test')
        print('')

        return y_true, y_pred

def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    warnings.filterwarnings("ignore", category=UserWarning) 

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')
    parser.add_argument('--config', type=str, required=True, help='config file path')
    # 新增超参数参数
    parser.add_argument('--vib_weight', type=float, default=None, help='VIB loss weight')
    parser.add_argument('--match_weight', type=float, default=None, help='Match loss weight')
    parser.add_argument('--contrastive_weight', type=float, default=None, help='Contrastive loss weight')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    set_random_seed(args.seed, use_cuda=True)

    with open(args.config) as config_file:
        config = json.load(config_file)
    
    # 如果命令行提供了超参数，覆盖config中的值
    if args.vib_weight is not None:
        config['vib_loss_weight'] = args.vib_weight
    if args.match_weight is not None:
        config['match_loss_weight'] = args.match_weight
    if args.contrastive_weight is not None:
        config['contrastive_loss_weight'] = args.contrastive_weight
    
    # 根据超参数生成实验名称
    base_name = os.path.basename(args.config).replace('.json', '')
    vib_w = config.get('vib_loss_weight', 1.0)
    match_w = config.get('match_loss_weight', 1.0)
    cont_w = config.get('contrastive_loss_weight', 1.0)
    
    # 生成带超参数的实验名称
    config['name'] = f"{base_name}_VIB{vib_w}_Match{match_w}_IMC{cont_w}"
    print(f"[INFO] Experiment name: {config['name']}")
    
    Y_true = np.zeros(0)
    Y_pred = np.zeros((0, config['classifier']['num_classes']))

    for fold in range(1, 11):
        trainer = OneFoldTrainer(args, fold, config)
        y_true, y_pred = trainer.run()
        Y_true = np.concatenate([Y_true, y_true])
        Y_pred = np.concatenate([Y_pred, y_pred])
    
        summarize_result(config, fold, Y_true, Y_pred)
    

if __name__ == "__main__":
    main()

# 使用方法示例:
# 默认配置（权重都为1.0）
# python train_M2M.py --config configs/Sleep-EDF-2018/M2M_2channel_fpzcz_h_SL-15.json --gpu 1

# 消融实验：调整VIB权重
# python train_M2M.py --config configs/Sleep-EDF-2018/M2M_2channel_fpzcz_h_SL-10.json --gpu 1 --vib_weight 0.5

# 消融实验：调整Match权重
# python train_M2M.py --config configs/Sleep-EDF-2018/M2M_2channel_fpzcz_h_SL-10.json --gpu 1 --match_weight 0.5

# 消融实验：同时调整多个权重
# python train_M2M.py --config configs/Sleep-EDF-2018/M2M_2channel_fpzcz_h_SL-10.json --vib_weight 1 --match_weight 1 --contrastive_weight 1 --gpu 1
# python train_M2M.py --config configs/Sleep-EDF-2018/M2M_3channel_fpzcz_h_pzoz_SL-10.json --vib_weight 1 --match_weight 0.5 --contrastive_weight 0.5 --gpu 1