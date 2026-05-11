import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, LayerNorm, Dropout
from torch import Tensor
from torch.nn import Linear

class CrossModalFusion(nn.Module):
    def __init__(self, d_model=64, nhead=8, dropout=0.1, dim_feedforward=256):
        super(CrossModalFusion, self).__init__()
        self.cross_attn = Cross_modal_atten(d_model=d_model, nhead=nhead, dropout=dropout)
        self.ffn = Feed_forward(d_model=d_model, dropout=dropout, dim_feedforward=dim_feedforward)

    def forward(self, eeg, eog):
        # eeg, eog: [B, T, C]
        fusion, attn_weights = self.cross_attn(eeg, eog)  # [B, T, C]
        out = self.ffn(fusion)  # [B, T, C]
        return out

def negative_pair_prob(sim_eeg2eog, sim_eog2eeg, eeg_feats, eog_feats, labels):
    """
    只根据负样本概率矩阵采样 EOG 负样本对。
    """
    batch_size = eeg_feats.shape[0]
    unique_labels = torch.unique(labels)
    if unique_labels.size(0) == 1:
        print("Warning: Only one class present in this batch, cannot sample negative pairs.")
        return None
        
    with torch.no_grad():
        labels_expand = labels.unsqueeze(0).expand(batch_size, batch_size)
        mask = (labels_expand == labels_expand.t()).to(sim_eeg2eog.device)
        sim_eeg2eog = sim_eeg2eog.clone()
        sim_eeg2eog.masked_fill_(mask, float('-inf'))
        prob_eeg2eog = torch.softmax(sim_eeg2eog, dim=1)  # [B, B]
        neg_idx_eog = [torch.multinomial(prob_eeg2eog[b], 1).item() for b in range(batch_size)]

    # 只采样 EOG 负样本
    eog_feats_neg = eog_feats[neg_idx_eog]  # [B, T, C]

    return eog_feats_neg


class AttnPool(nn.Module):
    """与classifiers.py中Transformer的attn pool一致"""
    def __init__(self, d_model):
        super().__init__()
        self.w_ha = nn.Linear(d_model, d_model, bias=True)
        self.w_at = nn.Linear(d_model, 1, bias=False)

    def forward(self, x):  # x: [B, T, C]
        a_states = torch.tanh(self.w_ha(x))  # [B, T, C]
        alpha = torch.softmax(self.w_at(a_states), dim=1)  # [B, T, 1]
        x = torch.sum(alpha * a_states, dim=1)  # [B, C]
        return x


class Matcher(nn.Module):
    def __init__(self, d_model=64, nhead=8, dropout=0.1, dim_feedforward=256):
        super(Matcher, self).__init__()
        self.fusion = CrossModalFusion(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            dim_feedforward=dim_feedforward
        )
        # 改为使用 Attention Pool
        self.pool = AttnPool(d_model)
        self.mlp_head = nn.Linear(d_model, 2)   # 分类头

    def forward(self, sim_eeg2eog, sim_eog2eeg, eeg_feats, eog_feats, labels, match=True):
        bs = eeg_feats.size(0)
        output_pos = self.fusion(eeg_feats, eog_feats)  # [B, T, C]
        #pooled_pos = self.pool(output_pos.transpose(1, 2)).squeeze(-1)  # [B, C] for AdaptiveAvgPool1d
        pooled_pos = self.pool(output_pos) # [B, C] for AttnPool
        if self.training and match:
            eog_feats_neg = negative_pair_prob(
                sim_eeg2eog, sim_eog2eeg, eeg_feats, eog_feats, labels
            )
            if eog_feats_neg is None:
                logits_output = self.mlp_head(pooled_pos)  # [B, 2]
                onehot_labels = torch.ones(bs, dtype=torch.long).to(pooled_pos.device)
                loss = F.cross_entropy(logits_output, onehot_labels)
            else:
                output_neg = self.fusion(eeg_feats, eog_feats_neg)  # [B, T, C]
                #pooled_neg = self.pool(output_neg.transpose(1, 2)).squeeze(-1)  # [B, C] for AdaptiveAvgPool1d
                pooled_neg = self.pool(output_neg) # [B, C] for AttnPool
                #pooled_neg =output_neg[:,0,:]  # [B, C], 直接取第一个token作为cls token
                
                cls_out = torch.cat([pooled_pos, pooled_neg], dim=0)    # [2B, C]
                logits_output = self.mlp_head(cls_out)                  # [2B, 2]

                onehot_labels = torch.cat([
                    torch.ones(bs, dtype=torch.long),
                    torch.zeros(bs, dtype=torch.long)
                ], dim=0).to(cls_out.device)

                loss = F.cross_entropy(logits_output, onehot_labels)
        else:
            logits_output = self.mlp_head(pooled_pos)  # [B, 2]
            onehot_labels = torch.ones(bs, dtype=torch.long).to(pooled_pos.device)
            loss = torch.tensor(0.0, device=pooled_pos.device, requires_grad=True)

        return output_pos, loss, pooled_pos
    

class Cross_modal_atten(nn.Module): 
    def __init__(self, d_model=64, nhead=8, dropout=0.1, layer_norm_eps=1e-5):
        super(Cross_modal_atten, self).__init__()
        self.norm = LayerNorm(d_model, eps=layer_norm_eps)  
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout = Dropout(dropout) 

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        attn_out, attn_weights = self.cross_attn(x1, x2, x2)
        out = x1 + self.dropout(attn_out)
        out = self.norm(out)
        return out, attn_weights


class Feed_forward(nn.Module): 
    def __init__(self, d_model=64, dropout=0.1, dim_feedforward=512, layer_norm_eps=1e-5):
        super(Feed_forward, self).__init__()
        self.norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.relu = nn.ReLU()
        self.dropout1 = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout2 = Dropout(dropout)
        
    def forward(self, x: Tensor) -> Tensor:        
        src = x
        src2 = self.linear2(self.dropout1(self.relu(self.linear1(src))))
        out = src + self.dropout2(src2)
        out = self.norm(out)
        return out
