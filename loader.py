import os
import glob
import torch
import numpy as np
from transform import *
from torch.utils.data import Dataset


class EEGDataLoader(Dataset):

    def __init__(self, config, fold, set_name='train'):
        self.set_name = set_name
        self.fold = fold
        self.sr = 100        
        self.dset_cfg = config['dataset']
        
        self.root_dir = self.dset_cfg['root_dir']
        self.dset_name = self.dset_cfg['name']
        self.num_splits = self.dset_cfg['num_splits']
        
        # 从配置中动态获取模态信息
        self.modalities = config.get('modalities', [])
        self.channel_mapping = self.dset_cfg.get('channel_mapping', {})
        assert set(self.modalities) == set(self.channel_mapping.keys()), "Modalities and channel_mapping keys must match."
        
        self.seq_len = self.dset_cfg['seq_len']
        self.target_idx = self.dset_cfg['target_idx']
        self.training_mode = config['training_params']['mode']

        self.dataset_path = os.path.join(self.root_dir, 'dset', self.dset_name, 'npz')
        
        # 加载所有模态的数据
        self.inputs, self.labels, self.epochs = self.split_dataset()
        
        if self.training_mode == 'pretrain':
            # (预训练逻辑保持不变，但请注意它可能需要适配多模态)
            self.transform = Compose(
                transforms=[
                    RandomAmplitudeScale(),
                    RandomTimeShift(),
                    RandomDCShift(),
                    RandomZeroMasking(),
                    RandomAdditiveGaussianNoise(),
                    RandomBandStopFilter(),
                ]
            )
            self.two_transform = TwoTransform(self.transform)

    def __len__(self):
        return len(self.epochs)

    def __getitem__(self, idx):
        n_sample = 30 * self.sr * self.seq_len
        file_idx, epoch_idx, seq_len = self.epochs[idx]
        
        # 准备一个字典来存放所有模态的输入
        inputs_dict = {}
        
        # 动态获取每个模态的数据
        for m in self.modalities:
            time_data = self.inputs[m][file_idx][epoch_idx : epoch_idx + seq_len]
            # 频域数据暂时不在此处使用，但可以按需添加
            # freq_data = self.inputs[m]['freq'][file_idx][epoch_idx : epoch_idx + seq_len]
            
            if self.set_name == 'train' and self.training_mode == 'pretrain':
                 # 预训练的数据增强逻辑
                assert seq_len == 1
                time_a, time_b = self.two_transform(time_data)
                inputs_dict[f'{m}_time'] = [torch.from_numpy(time_a).float(), torch.from_numpy(time_b).float()]
            else:
                # 微调或测试时的逻辑
                time_data = time_data.reshape(1, n_sample)
                inputs_dict[f'{m}_time'] = torch.from_numpy(time_data).float()

        labels = self.labels[file_idx][epoch_idx : epoch_idx + seq_len]
        labels = torch.from_numpy(labels).long()
        labels = labels[self.target_idx]
        
        return inputs_dict, labels

    def split_dataset(self):
        # 只存储时域数据
        inputs = {m: [] for m in self.modalities}
        labels, epochs = [], []
        
        # 确定主模态用于查找文件列表和标签（假设所有通道文件一一对应）
        main_modality_id = self.modalities[0]
        main_channel_name = self.channel_mapping[main_modality_id]
        main_data_root = os.path.join(self.dataset_path, main_channel_name)
        
        data_fname_list = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(main_data_root, '*.npz')))]
        
        # 数据集划分逻辑 (保持不变)
        data_fname_dict = {'train': [], 'test': [], 'val': []}
        split_idx_list = np.load(os.path.join('./split_idx', 'idx_{}.npy'.format(self.dset_name)), allow_pickle=True)
        assert len(split_idx_list) == self.num_splits
        if self.dset_name == 'Sleep-EDF-2018':
            for i in range(len(data_fname_list)):
                subject_idx = int(data_fname_list[i][3:5])
                if subject_idx in split_idx_list[self.fold - 1][self.set_name]:
                    data_fname_dict[self.set_name].append(data_fname_list[i])
        # ... (其他数据集的划分逻辑)
        else:
            raise NameError("dataset '{}' cannot be found.".format(self.dset_name))
            
        file_idx = 0
        for data_fname in data_fname_dict[self.set_name]:
            # 加载主模态的标签
            main_npz_file = np.load(os.path.join(main_data_root, data_fname))
            labels.append(main_npz_file['y'])
            
            # 遍历所有模态并加载数据
            for m in self.modalities:
                channel_name = self.channel_mapping[m]
                
                # 加载时域数据
                time_data_root = os.path.join(self.dataset_path, channel_name)
                time_npz_file = np.load(os.path.join(time_data_root, data_fname))
                inputs[m].append(time_npz_file['x'])

            # 创建epoch索引 (保持不变)
            seq_len = self.seq_len
            for i in range(len(main_npz_file['y']) - seq_len + 1):
                epochs.append([file_idx, i, seq_len])
            file_idx += 1
        
        print(f"[INFO] Loaded data for {self.set_name} set, {len(self.modalities)} modalities:")
        for m in self.modalities:
            print(f"  - Modality '{m}' ({self.channel_mapping[m]}): {len(inputs[m])} files")
        
        return inputs, labels, epochs
