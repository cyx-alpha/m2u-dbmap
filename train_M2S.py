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
from models.model_M2S import M2SModel as MainModel


class OneFoldTrainer:
    def __init__(self, args, fold, config):
        self.args = args
        self.fold = fold
        
        self.cfg = config
        self.ds_cfg = config['dataset']
        self.tp_cfg = config['training_params']
        self.es_cfg = self.tp_cfg['early_stopping']
        
        # The single modality being trained
        self.modality = self.cfg['modality']
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('[INFO] Config name: {}'.format(config['name']))

        self.train_iter = 0
        self.model = self.build_model()
        self.loader_dict = self.build_dataloader()
        
        self.criterion = nn.CrossEntropyLoss()
        self.activate_train_mode()
        self.optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=self.tp_cfg['lr'], weight_decay=self.tp_cfg['weight_decay'])
        
        self.ckpt_path = os.path.join('checkpoints', config['name'])
        os.makedirs(self.ckpt_path, exist_ok=True)
        self.ckpt_name = 'ckpt_fold-{0:02d}.pth'.format(self.fold)
        self.early_stopping = EarlyStopping(patience=self.es_cfg['patience'], verbose=True, ckpt_path=self.ckpt_path, ckpt_name=self.ckpt_name, mode=self.es_cfg['mode'])
        

    def build_model(self):
        model = MainModel(self.cfg)
        print('[INFO] Number of params of model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
        model = torch.nn.DataParallel(model, device_ids=list(range(len(self.args.gpu.split(",")))))
        
        if self.tp_cfg['mode'] != 'scratch':
            print('[INFO] Model loaded for M2S finetune from multimodal joint training')
            self.load_multimodal_joint_weights(model)

        model.to(self.device)
        print('[INFO] Model prepared, Device used: {} GPU:{}'.format(self.device, self.args.gpu))
        return model

    def load_multimodal_joint_weights(self, model):
        """Load weights from a jointly trained multi-modal model."""
        joint_training_source = self.cfg.get('joint_training_source')
        if not joint_training_source:
            print('[WARNING] `joint_training_source` not specified in config. Training from scratch.')
            return

        # 构建带超参数的联合训练源路径
        vib_w = self.cfg.get('vib_loss_weight', 1.0)
        match_w = self.cfg.get('match_loss_weight', 1.0)
        cont_w = self.cfg.get('contrastive_loss_weight', 1.0)
        
        # 生成联合训练源的完整名称（带超参数）
        joint_training_full_name = f"{joint_training_source}_VIB{vib_w}_Match{match_w}_IMC{cont_w}"
        joint_training_path = os.path.join('./checkpoints', joint_training_full_name, 'ckpt_fold-{0:02d}.pth'.format(self.fold))
        
        print(f'[INFO] Loading multimodal joint training weights from: {joint_training_path}')
        
        if os.path.exists(joint_training_path):
            joint_state_dict = torch.load(joint_training_path, map_location=self.device)
            
            # Create a new state_dict for the M2S model
            m2s_state_dict = {}
            m = self.modality
            
            for key, value in joint_state_dict.items():
                # Map feature_single.{modality} -> feature_single
                if key.startswith(f'module.feature_single.{m}'):
                    new_key = key.replace(f'module.feature_single.{m}', 'module.feature_single')
                    m2s_state_dict[new_key] = value
                # Map feature_multi.{modality} -> feature_multi
                elif key.startswith(f'module.feature_multi.{m}'):
                    new_key = key.replace(f'module.feature_multi.{m}', 'module.feature_multi')
                    m2s_state_dict[new_key] = value
                # Map classifier_single.{modality} -> classifier_single
                elif key.startswith(f'module.classifier_single.{m}'):
                    new_key = key.replace(f'module.classifier_single.{m}', 'module.classifier_single')
                    m2s_state_dict[new_key] = value
                # Map classifier_multi.{modality} -> classifier_multi
                elif key.startswith(f'module.classifier_multi.{m}'):
                    new_key = key.replace(f'module.classifier_multi.{m}', 'module.classifier_multi')
                    m2s_state_dict[new_key] = value
                # Map combined_vib.{modality} -> combined_vib
                elif key.startswith(f'module.combined_vib.{m}'):
                    new_key = key.replace(f'module.combined_vib.{m}', 'module.combined_vib')
                    m2s_state_dict[new_key] = value

            model.load_state_dict(m2s_state_dict, strict=False)
            print(f'[INFO] Loaded {len(m2s_state_dict)} weight tensors for modality "{m}" successfully.')
        else:
            print(f'[WARNING] Multimodal joint training weights not found at {joint_training_path}. Training from scratch.')
        
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
            print('[INFO] Freezing feature extractors and transformer encoders for M2S fine-tuning')
            
            # ===== 1. 冻结 Single-path 的特征提取部分 =====
            # 冻结 CNN backbone
            self.model.module.feature_single.train(False)
            for p in self.model.module.feature_single.parameters():
                p.requires_grad = False
            print('[INFO] Single-path CNN backbone frozen')
            
            # 冻结 Single-path classifier 的 transformer 部分
            classifier_single = self.model.module.classifier_single
            
            # 冻结位置编码
            if hasattr(classifier_single, 'pos_encoding'):
                classifier_single.pos_encoding.train(False)
                for p in classifier_single.pos_encoding.parameters():
                    p.requires_grad = False
                print('[INFO] Single-path positional encoding frozen')
            
            # 冻结 Transformer 编码器
            if hasattr(classifier_single, 'transformer'):
                classifier_single.transformer.train(False)
                for p in classifier_single.transformer.parameters():
                    p.requires_grad = False
                print('[INFO] Single-path transformer encoder frozen')
            
            # 保持 Single-path attention pooling 可训练
            if hasattr(classifier_single, 'w_ha'):
                classifier_single.w_ha.train(True)
                for p in classifier_single.w_ha.parameters():
                    p.requires_grad = True
                print('[INFO] Single-path attention pooling w_ha kept trainable')
            
            if hasattr(classifier_single, 'w_at'):
                classifier_single.w_at.train(True)
                for p in classifier_single.w_at.parameters():
                    p.requires_grad = True
                print('[INFO] Single-path attention pooling w_at kept trainable')
            
            # 冻结 Single-path 原始 fc 层
            if hasattr(classifier_single, 'fc'):
                classifier_single.fc.train(False)
                for p in classifier_single.fc.parameters():
                    p.requires_grad = False
                print('[INFO] Single-path original fc layer frozen')
            
            # ===== 2. 冻结 Multi-path 的特征提取部分 =====
            # 冻结 CNN backbone
            self.model.module.feature_multi.train(False)
            for p in self.model.module.feature_multi.parameters():
                p.requires_grad = False
            print('[INFO] Multi-path CNN backbone frozen')
            
            # 冻结 Multi-path classifier 的 transformer 部分
            classifier_multi = self.model.module.classifier_multi
            
            # 冻结位置编码
            if hasattr(classifier_multi, 'pos_encoding'):
                classifier_multi.pos_encoding.train(False)
                for p in classifier_multi.pos_encoding.parameters():
                    p.requires_grad = False
                print('[INFO] Multi-path positional encoding frozen')
            
            # 冻结 Transformer 编码器
            if hasattr(classifier_multi, 'transformer'):
                classifier_multi.transformer.train(False)
                for p in classifier_multi.transformer.parameters():
                    p.requires_grad = False
                print('[INFO] Multi-path transformer encoder frozen')
            
            # 保持 Multi-path attention pooling 可训练
            if hasattr(classifier_multi, 'w_ha'):
                classifier_multi.w_ha.train(True)
                for p in classifier_multi.w_ha.parameters():
                    p.requires_grad = True
                print('[INFO] Multi-path attention pooling w_ha kept trainable')
            
            if hasattr(classifier_multi, 'w_at'):
                classifier_multi.w_at.train(True)
                for p in classifier_multi.w_at.parameters():
                    p.requires_grad = True
                print('[INFO] Multi-path attention pooling w_at kept trainable')
            
            # 冻结 Multi-path 原始 fc 层
            if hasattr(classifier_multi, 'fc'):
                classifier_multi.fc.train(False)
                for p in classifier_multi.fc.parameters():
                    p.requires_grad = False
                print('[INFO] Multi-path original fc layer frozen')

            # ===== 3. 保持可训练的部分 =====
            # VIB 和 final_classifier 保持可训练
            self.model.module.combined_vib.train(True)
            for p in self.model.module.combined_vib.parameters():
                p.requires_grad = True
            
            self.model.module.final_classifier.train(True)
            for p in self.model.module.final_classifier.parameters():
                p.requires_grad = True
            
            print('[INFO] Summary: CNN backbones + Transformer encoders frozen')
            print('[INFO]          Attention pooling layers + VIB + final_classifier trainable')

    def _prepare_inputs(self, inputs):
        """Prepare input for the single modality."""
        processed_inputs = {}
        # The model only needs the time-domain signal for the specified modality
        processed_inputs[f'{self.modality}_time'] = inputs[f'{self.modality}_time'].to(self.device)
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
    parser.add_argument('--config', type=str, help='config file path')
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

    for fold in range(1,11):
        trainer = OneFoldTrainer(args, fold, config)
        y_true, y_pred = trainer.run()
        Y_true = np.concatenate([Y_true, y_true])
        Y_pred = np.concatenate([Y_pred, y_pred])
    
        summarize_result(config, fold, Y_true, Y_pred)
    

if __name__ == "__main__":
    main()

# 使用方法示例:
# 默认配置（使用config中的权重值，默认都为1.0）
# python train_M2S.py --config configs/Sleep-EDF-2018/M2S_Fpz-Cz_from_2channel_fpzcz_h_SL-08.json --gpu 0

# 消融实验：调整VIB权重
# python train_M2S.py --config configs/Sleep-EDF-2018/M2S_Fpz-Cz_from_2channel_fpzcz_h_SL-10.json --gpu 1 --vib_weight 0.5

# 消融实验：调整Match权重
# python train_M2S.py --config configs/Sleep-EDF-2018/M2S_Fpz-Cz_from_2channel_fpzcz_h_SL-10.json --gpu 1 --match_weight 0.5

# 消融实验：同时调整多个权重
# python train_M2S.py --config configs/Sleep-EDF-2018/M2S_Fpz-Cz_from_2channel_fpzcz_h_SL-10.json --vib_weight 0.5 --match_weight 1 --contrastive_weight 1 --gpu 1
# python train_M2S.py --config configs/Sleep-EDF-2018/M2S_Fpz-Cz_from_3channel_fpzcz_h_pzoz_SL-10.json --vib_weight 1 --match_weight 0.5 --contrastive_weight 0.5 --gpu 1
# 2. 使用相同的超参数训练M2S模型（会自动加载对应的M2M权重）
# python train_M2S.py --config configs/Sleep-EDF-2018/M2S_horizontal_from_2channel_fpzcz_h_SL-10.json --vib_weight 1 --match_weight 1 --contrastive_weight 1 --gpu 1