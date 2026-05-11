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
from models.model_SingleBranch import SingleBranchModel as MainModel


class OneFoldTrainer:
    def __init__(self, args, fold, config, branch_type):
        self.args = args
        self.fold = fold
        self.branch_type = branch_type  # 'single' or 'multi'
        
        self.cfg = config
        self.ds_cfg = config['dataset']
        self.tp_cfg = config['training_params']
        self.es_cfg = self.tp_cfg['early_stopping']
        
        # 获取单一模态
        self.modalities = self.cfg.get('modalities', [])
        assert len(self.modalities) == 1, "SingleBranch trainer only supports single modality"
        self.modality = self.modalities[0]
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[INFO] Config name: {config["name"]} - Branch: {branch_type}')

        self.train_iter = 0
        self.model = self.build_model()
        self.loader_dict = self.build_dataloader()
        
        self.criterion = nn.CrossEntropyLoss()
        self.activate_train_mode()
        self.optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad], 
                                    lr=self.tp_cfg['lr'], 
                                    weight_decay=self.tp_cfg['weight_decay'])
        
        # 修改checkpoint路径，区分single和multi分支
        base_ckpt_name = config['name']
        self.ckpt_path = os.path.join('checkpoints', f'{base_ckpt_name}_{branch_type}_branch')
        os.makedirs(self.ckpt_path, exist_ok=True)
        self.ckpt_name = 'ckpt_fold-{0:02d}.pth'.format(self.fold)
        self.early_stopping = EarlyStopping(patience=self.es_cfg['patience'], 
                                           verbose=True, 
                                           ckpt_path=self.ckpt_path, 
                                           ckpt_name=self.ckpt_name, 
                                           mode=self.es_cfg['mode'])

    def build_model(self):
        model = MainModel(self.cfg, branch_type=self.branch_type)
        print('[INFO] Number of params of model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
        model = torch.nn.DataParallel(model, device_ids=list(range(len(self.args.gpu.split(",")))))
        
        # 加载预训练权重
        if self.tp_cfg['mode'] != 'scratch':
            print(f'[INFO] Loading {self.branch_type} branch weights')
            self.load_branch_weights(model)

        model.to(self.device)
        print('[INFO] Model prepared, Device used: {} GPU:{}'.format(self.device, self.args.gpu))
        return model

    def load_branch_weights(self, model):
        """
        智能加载权重：自动检测源模型是 M2S 还是 M2M
        """
        source_config = self.cfg.get('source_model')
        if not source_config:
            print('[WARNING] `source_model` not specified in config. Training from scratch.')
            return

        source_path = os.path.join('./checkpoints', source_config, 'ckpt_fold-{0:02d}.pth'.format(self.fold))
        
        print(f'[INFO] Loading {self.branch_type} branch weights from: {source_path}')
        
        if not os.path.exists(source_path):
            print(f'[WARNING] Source weights not found at {source_path}. Training from scratch.')
            return
            
        source_state_dict = torch.load(source_path, map_location=self.device)
        
        # 创建新的state_dict用于当前模型
        branch_state_dict = {}
        
        # 智能检测源模型类型并映射权重
        # 检查是否存在 M2S 特征（不带通道名）
        is_m2s = any(key.startswith('module.feature_single.') and 
                    not any(f'.{m}.' in key for m in ['eeg_fpzcz', 'eeg_pzoz', 'eog_h']) 
                    for key in source_state_dict.keys())
        
        if is_m2s:
            print('[INFO] Detected M2S model structure (no modality names in keys)')
            # M2S 模型：feature_single -> feature_extractor
            source_feature_prefix = f'module.feature_{self.branch_type}.'
            source_classifier_prefix = f'module.classifier_{self.branch_type}.'
        else:
            print(f'[INFO] Detected M2M model structure (modality: {self.modality})')
            # M2M 模型：feature_single.{modality} -> feature_extractor
            source_feature_prefix = f'module.feature_{self.branch_type}.{self.modality}.'
            source_classifier_prefix = f'module.classifier_{self.branch_type}.{self.modality}.'
        
        # 映射权重
        for key, value in source_state_dict.items():
            new_key = None
            
            # 映射特征提取器
            if key.startswith(source_feature_prefix):
                new_key = key.replace(source_feature_prefix, 'module.feature_extractor.')
            
            # 映射分类器
            elif key.startswith(source_classifier_prefix):
                new_key = key.replace(source_classifier_prefix, 'module.classifier.')
            
            if new_key:
                branch_state_dict[new_key] = value
        
        # 加载权重
        missing_keys, unexpected_keys = model.load_state_dict(branch_state_dict, strict=False)
        
        print(f'[INFO] Loaded {len(branch_state_dict)} weight tensors for {self.branch_type} branch')
        
        if missing_keys:
            # 过滤掉 linear_head 的 missing keys（这是预期的，因为它需要重新训练）
            filtered_missing = [k for k in missing_keys if 'linear_head' not in k]
            if filtered_missing:
                print(f'[WARNING] Missing keys (excluding linear_head): {filtered_missing}')
        
        if unexpected_keys:
            print(f'[WARNING] Unexpected keys: {unexpected_keys}')
        
    def build_dataloader(self):
        train_dataset = EEGDataLoader(self.cfg, self.fold, set_name='train')
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.tp_cfg['batch_size'], 
                                 shuffle=True, num_workers=4*len(self.args.gpu.split(",")), pin_memory=True)
        val_dataset = EEGDataLoader(self.cfg, self.fold, set_name='val')
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.tp_cfg['batch_size'], 
                               shuffle=False, num_workers=4*len(self.args.gpu.split(",")), pin_memory=True)
        test_dataset = EEGDataLoader(self.cfg, self.fold, set_name='test')
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.tp_cfg['batch_size'], 
                                shuffle=False, num_workers=4*len(self.args.gpu.split(",")), pin_memory=True)
        print('[INFO] Dataloader prepared')

        return {'train': train_loader, 'val': val_loader, 'test': test_loader}
    
    def activate_train_mode(self):
        self.model.train()
        if self.tp_cfg['mode'] == 'freezefinetune':
            print(f'[INFO] Freezing only feature extractor for {self.branch_type} branch fine-tuning')
            
            # 1. 冻结特征提取器 (CNN backbone)
            self.model.module.feature_extractor.train(False)
            for p in self.model.module.feature_extractor.parameters():
                p.requires_grad = False
            print('[INFO] Feature extractor (CNN backbone) frozen')
            
            # 2. 解冻整个 Classifier（包括 Transformer 和 Attention Pooling）
            self.model.module.classifier.train(True)
            for p in self.model.module.classifier.parameters():
                p.requires_grad = True
            print('[INFO] ✓ Entire classifier (Transformer + Attention Pooling + fc) unfrozen')
            
            # 3. 保持 linear_head 可训练
            self.model.module.linear_head.train(True)
            for p in self.model.module.linear_head.parameters():
                p.requires_grad = True
            print('[INFO] Linear head kept trainable')
            
            # ===== 打印详细的参数解冻状态 =====
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f'\n[INFO] Trainable params: {trainable_params:,}/{total_params:,} ({100*trainable_params/total_params:.2f}%)')
            
            # 详细列出可训练参数的模块
            print('\n[INFO] Trainable modules:')
            for name, module in self.model.named_modules():
                if any(p.requires_grad for p in module.parameters()):
                    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                    if num_params > 0:
                        print(f'  ✓ {name}: {num_params:,} params')
            
            print('\n[INFO] Summary: Only CNN frozen, Classifier (Transformer + Attention + fc) + linear_head trainable')

    def _prepare_inputs(self, inputs):
        """Prepare input for the single modality."""
        processed_inputs = {}
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
            print(f'\n[INFO] Fold: {self.fold}, Epoch: {epoch}, Branch: {self.branch_type}')
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
    parser.add_argument('--branch', type=str, default='single', choices=['single', 'multi'], 
                       help='which branch to train: single or multi')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    set_random_seed(args.seed, use_cuda=True)

    with open(args.config) as config_file:
        config = json.load(config_file)
    config['name'] = os.path.basename(args.config).replace('.json', '')
    
    print(f'\n[INFO] ========== Training {args.branch.upper()} Branch ==========\n')
    
    Y_true = np.zeros(0)
    Y_pred = np.zeros((0, config['classifier']['num_classes']))

    for fold in range(1,11):
        trainer = OneFoldTrainer(args, fold, config, branch_type=args.branch)
        y_true, y_pred = trainer.run()
        Y_true = np.concatenate([Y_true, y_true])
        Y_pred = np.concatenate([Y_pred, y_pred])
    
        summarize_result(config, fold, Y_true, Y_pred)
    

if __name__ == "__main__":
    main()

# 使用示例:
# 从 M2S 模型加载:
# python train_SingleBranch.py --config configs/Sleep-EDF-2018/SingleBranch_Fpz-Cz_from_M2S_SL-10.json --gpu 0 --branch single

# 从 M2M 模型加载:
# python train_SingleBranch.py --config configs/Sleep-EDF-2018/SingleBranch_Fpz-Cz_from_M2M_SL-10.json --gpu 0 --branch single
# python train_SingleBranch.py --config configs/Sleep-EDF-2018/SingleBranch_Fpz-Cz_from_M2M_SL-10_IMC0.json --gpu 0 --branch multi
# python train_SingleBranch.py --config configs/Sleep-EDF-2018/SingleBranch_Horizontal_from_M2M_SL-10.json --branch single --gpu 0
# python train_SingleBranch.py --config configs/Sleep-EDF-2018/SingleBranch_Horizontal_from_M2M_SL-10.json --branch multi --gpu 0