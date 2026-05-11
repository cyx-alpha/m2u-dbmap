import torch.nn as nn
import torch.nn.functional as F
import torch
from .sleepyco import SleePyCoBackbone
from .classifiers import get_classifier
from .VIB import VIB
from .infoNCE_loss import info_nce_loss
from .match import Matcher
import itertools


class MainModel(nn.Module):
    
    def __init__(self, config):
        super(MainModel, self).__init__()

        self.cfg = config
        self.training_mode = config['training_params']['mode']
        
        self.modalities = config.get('modalities', ['eeg', 'eog'])

        self.num_classes = config['classifier']['num_classes']
        
        # 读取消融实验控制开关
        self.match_enable = config.get('match_enable', True)  # 默认启用 match
        
        # 读取损失权重（默认都为1.0）
        self.vib_loss_weight = config.get('vib_loss_weight', 1.0)
        self.match_loss_weight = config.get('match_loss_weight', 1.0)
        self.contrastive_loss_weight = config.get('contrastive_loss_weight', 1.0)

        if self.training_mode in ['scratch', 'fullfinetune', 'freezefinetune']:
            self.feature_single = nn.ModuleDict()
            self.feature_multi = nn.ModuleDict()
            self.classifier_single = nn.ModuleDict()
            self.classifier_multi = nn.ModuleDict()
            
            # ==== NEW: per-branch (front) VIBs for single & multi branches ====
            # 假设 pooled 特征维度为 128（与原 combined_vib 中使用的一致）
            self.vib_single_branch = nn.ModuleDict()
            # 删除：self.vib_multi_branch = nn.ModuleDict()
            # 新增：对交互后的特征做VIB
            self.vib_interactive = nn.ModuleDict()

            for m in self.modalities:
                self.feature_single[m] = SleePyCoBackbone(self.cfg)
                self.feature_multi[m] = SleePyCoBackbone(self.cfg)
                self.classifier_single[m] = get_classifier(config)
                self.classifier_multi[m] = get_classifier(config)

                # 对每个分支在"前面"各自做一次 VIB（直接输入 pooled token）
                self.vib_single_branch[m] = VIB(d_x=128, d_z=128, n_classes=self.num_classes)
                # 删除：self.vib_multi_branch[m] = VIB(d_x=128, d_z=128, n_classes=self.num_classes)
                # 新增：对交互后的特征做VIB
                self.vib_interactive[m] = VIB(d_x=128, d_z=128, n_classes=self.num_classes)

            # ========== 修正：为每个有向模态对创建独立的Matcher ==========
            self.matchers = nn.ModuleDict()
            for m1 in self.modalities:
                for m2 in self.modalities:
                    if m1 != m2:
                        # 有向匹配: m1 -> m2
                        matcher_key = f"{m1}_to_{m2}"
                        self.matchers[matcher_key] = Matcher(d_model=128, nhead=8, dropout=0.1)

            # ==== NEW: 为每个有向交互创建门控网络 ====
            self.interaction_gates = nn.ModuleDict()
            for m_query in self.modalities:
                for m_kv in self.modalities:
                    if m_query != m_kv:
                        gate_key = f"{m_query}_gate_{m_kv}"
                        # 输入: matcher输出的交互特征 (128维), 输出: 标量权重
                        self.interaction_gates[gate_key] = nn.Linear(128, 1)

            # 原有：两个分支拼接后的 VIB 保持不变
            self.combined_vib = nn.ModuleDict()
            for m in self.modalities:
                self.combined_vib[m] = VIB(d_x=128 * 2, d_z=128, n_classes=self.num_classes)
            
            self.final_mlp = nn.Linear(len(self.modalities) * 128, self.num_classes)

    def forward(self, x, labels=None):

        single_pooled, multi_transformer_features, multi_pooled = {}, {}, {}
        # 1. 特征提取
        for m in self.modalities:
            # single 分支
            single_features = self.feature_single[m](x[f'{m}_time'])
            _, _, single_pooled[m] = self.classifier_single[m](single_features[0].transpose(1, 2))

            # multi 分支
            multi_features = self.feature_multi[m](x[f'{m}_time'])
            _, multi_transformer_features[m], multi_pooled[m] = self.classifier_multi[m](multi_features[0].transpose(1, 2))

        # 1.1 每个分支前置 VIB
        total_vib_loss = 0.0
        single_vib_tokens = {}  # 存储VIB处理后的single特征
        
        for m in self.modalities:
            # Single-branch front VIB - 保存处理后的token
            single_vib_tokens[m], vib_loss_single, _ = self.vib_single_branch[m](single_pooled[m], labels)
            total_vib_loss += vib_loss_single.mean()

            # 删除：Multi-branch front VIB
            # _, vib_loss_multi, _ = self.vib_multi_branch[m](multi_pooled[m], labels)
            # total_vib_loss += vib_loss_multi.mean()

        # 2. 并行多模态交互
        total_contrastive_loss = 0.0
        total_match_loss = 0.0
        
        # --- 对比损失 (两两组合,无向) ---
        for m1, m2 in itertools.combinations(self.modalities, 2):
            contrastive_loss, _, _ = info_nce_loss(multi_pooled[m1], multi_pooled[m2])
            total_contrastive_loss += contrastive_loss
        # --- 匹配任务 (有向,每个 query 模态与其他所有 kv 模态交互) ---
        final_interactive_tokens = {}
        num_match_ops = 0
        
        for m_query in self.modalities:
            interactive_features_for_current_query = []
            gate_logits = []  # 存储门控 logits
            
            # 如果不启用 match，直接使用自己的 multi_pooled 作为交互特征
            if not self.match_enable:
                final_interactive_tokens[m_query] = multi_pooled[m_query]
                
                continue
            
            for m_kv in self.modalities:
                if m_query == m_kv:
                    continue
                num_match_ops += 1
                
                # ========== 使用专属的有向 matcher ==========
                matcher_key = f"{m_query}_to_{m_kv}"
                
                # 相似度矩阵（query -> kv 与 kv -> query）
                sim_query_to_kv = multi_pooled[m_query] @ multi_pooled[m_kv].t()
                sim_kv_to_query = multi_pooled[m_kv] @ multi_pooled[m_query].t()

                # 调用专属 matcher（注意参数映射）
                _, match_loss, pooled_feature = self.matchers[matcher_key](
                    sim_eeg2eog=sim_query_to_kv,  # query -> kv 的相似度
                    sim_eog2eeg=sim_kv_to_query,  # kv -> query 的相似度
                    eeg_feats=multi_transformer_features[m_query],  # query 的 transformer 特征
                    eog_feats=multi_transformer_features[m_kv],     # kv 的 transformer 特征
                    labels=labels,
                    match=True  # match_enable=True 时才会进入这个分支
                )
                
                total_match_loss += match_loss
                interactive_features_for_current_query.append(pooled_feature)
                
                # ==== 对 matcher 输出的交互特征计算门控权重 ====
                gate_key = f"{m_query}_gate_{m_kv}"
                gate_logit = self.interaction_gates[gate_key](pooled_feature)  # [batch, 1]
                gate_logits.append(gate_logit)

            # 对 query 模态从所有其他模态获得的特征进行动态加权平均
            if not interactive_features_for_current_query:
                # 只有一个模态的情况
                final_interactive_tokens[m_query] = multi_pooled[m_query]
            else:

                # 计算 softmax 权重
                stacked_features = torch.stack(interactive_features_for_current_query, dim=0)  # [num_kv, batch, 128]
                stacked_logits = torch.cat(gate_logits, dim=1)  # [batch, num_kv]
                gate_weights = F.softmax(stacked_logits, dim=1)  # [batch, num_kv]
                
                # 加权求和: gate_weights [batch, num_kv] -> [num_kv, batch, 1]
                gate_weights = gate_weights.t().unsqueeze(-1)  # [num_kv, batch, 1]
                weighted_features = stacked_features * gate_weights  # [num_kv, batch, 128]
                final_interactive_tokens[m_query] = torch.sum(weighted_features, dim=0)  # [batch, 128]

        # 3. 特征融合与最终分类
        final_tokens = []
        
        for m in self.modalities:
            # 先对交互后的特征做VIB
            interactive_token, vib_loss_interactive, _ = self.vib_interactive[m](
                final_interactive_tokens[m], labels
            )
            total_vib_loss += vib_loss_interactive.mean()
            
            # 拼接 single 分支VIB后的token 与 交互特征VIB后的token
            combined_feature = torch.cat([single_vib_tokens[m], interactive_token], dim=1)
            final_token, vib_loss_after_concat, _ = self.combined_vib[m](combined_feature, labels)
            final_tokens.append(final_token)
            total_vib_loss += vib_loss_after_concat.mean()

        final_feature_combined = torch.cat(final_tokens, dim=1)
        cls_output = self.final_mlp(final_feature_combined)
        final_classification_loss = F.cross_entropy(cls_output, labels)
        total_loss = (
            final_classification_loss
            + self.vib_loss_weight * total_vib_loss
            + self.contrastive_loss_weight * total_contrastive_loss
            + self.match_loss_weight * total_match_loss
        )
        
        if self.training:
            return cls_output, total_loss
        else:
            return cls_output, final_classification_loss
