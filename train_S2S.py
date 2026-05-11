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
from models.model_S2S import MainModel


class OneFoldTrainer:
    def __init__(self, args, fold, config):
        self.args = args
        self.fold = fold
        
        self.cfg = config
        self.ds_cfg = config['dataset']
        self.tp_cfg = config['training_params']
        self.es_cfg = self.tp_cfg['early_stopping']
        
        # 获取单模态的模态名和通道名
        self.modalities = self.cfg['modalities']
        assert len(self.modalities) == 1, "S2S model expects exactly one modality in config."
        self.modality_name = self.modalities[0]
        self.channel = self.ds_cfg['channel_mapping'][self.modality_name]
        
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
            print('[INFO] Model loaded for single-modal finetune')
            self.load_pretrain_weights(model)

        model.to(self.device)
        print('[INFO] Model prepared, Device used: {} GPU:{}'.format(self.device, self.args.gpu))
        return model

    def load_pretrain_weights(self, model):
        """为单模态加载预训练权重"""
        pretrain_root_path = os.path.join('checkpoints', self.ds_cfg['name'], 'single_pretrain')
        channel_folder = self.channel + '_pretrain'
        
        load_path = os.path.join(pretrain_root_path, channel_folder, 'ckpt_fold-{0:02d}.pth'.format(self.fold))
        print(f'[INFO] Loading pretrain weights for channel "{self.channel}" from: {load_path}')

        if os.path.exists(load_path):
            state_dict = torch.load(load_path, map_location=self.device)
            filtered_dict = {}
            for k, v in state_dict.items():
                # 将预训练模型中的 'eeg_feature' 键名替换为当前模型的 'feature_extractor'
                if 'feature' in k:
                    new_key = k.replace('module.feature', 'module.feature_extractor')
                    filtered_dict[new_key] = v
            
            model.load_state_dict(filtered_dict, strict=False)
            print(f'[INFO] Pretrained weights for "{self.channel}" loaded successfully')
        else:
            print(f'[WARNING] Pretrained weights for channel "{self.channel}" not found at {load_path}')
        
    def build_dataloader(self):
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
        if self.tp_cfg['mode'] == 'freezefinetune':
            print('[INFO] Freezing backbone')
            network = self.model.module.feature_extractor
            network.train(False)
            for p in network.parameters():
                p.requires_grad = False
            print(f'[INFO] feature_extractor backbone frozen')

            # 解冻C3, C4, C5层进行微调
            for conv_name in ['conv_c3', 'conv_c4', 'conv_c5']:
                if hasattr(network, conv_name):
                    conv_layer = getattr(network, conv_name)
                    conv_layer.train(True)
                    for p in conv_layer.parameters():
                        p.requires_grad = True
                    print(f'[INFO] feature_extractor.{conv_name} unfrozen for finetuning')

    def _prepare_inputs(self, inputs):
        """准备单模态的输入数据"""
        # S2S模型只使用一个输入，即对应模态的时域信号
        # loader返回的key是基于模态名，而不是通道名
        return inputs[f'{self.modality_name}_time'].to(self.device)

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
    parser.add_argument('--config', type=str, help='config file path')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    set_random_seed(args.seed, use_cuda=True)

    with open(args.config) as config_file:
        config = json.load(config_file)
    config['name'] = os.path.basename(args.config).replace('.json', '')
    
    Y_true = np.zeros(0)
    Y_pred = np.zeros((0, config['classifier']['num_classes']))

    for fold in range(1, config['dataset']['num_splits'] + 1):
        trainer = OneFoldTrainer(args, fold, config)
        y_true, y_pred = trainer.run()
        Y_true = np.concatenate([Y_true, y_true])
        Y_pred = np.concatenate([Y_pred, y_pred])
    
        summarize_result(config, fold, Y_true, Y_pred)
    

if __name__ == "__main__":
    main()

# python train_S2S.py --config configs/Sleep-EDF-2018/S2S_SL-08_channel-Fpz-Cz.json --gpu 0
# python train_S2S.py --config configs/Sleep-EDF-2018/S2S_SL-10_channel-Pz-Oz.json --gpu 0
# python train_S2S.py --config configs/Sleep-EDF-2018/S2S_SL-10_channel-horizontal.json --gpu 0
#python train_S2S.py --config configs/Sleep-EDF-2018/S2S_SL-10_channel-submental.json --gpu 0

# python train_S2S.py --config configs/Sleep-EDF-2018/S2S_SL-10_channel-Fpz-Cz.json --gpu 0
#python train_S2S.py --config configs/Sleep-EDF-2018/S2S_SL-10_channel-Pz-Oz.json --gpu 0
#python train_S2S.py --config configs/Sleep-EDF-2018/S2S_SL-10_channel-horizontal_scratch.json --gpu 1