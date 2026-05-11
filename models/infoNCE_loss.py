import torch
import torch.nn.functional as F

def info_nce_loss(eeg: torch.Tensor,
                  eog: torch.Tensor,
                  temperature: float = 0.1) -> torch.Tensor:
    """
    计算双向 InfoNCE 损失，实现对两组 [batch, dim] 表征的对比学习。
    
    Args:
        x:       Tensor, shape [N, D]，第一组表征。
        x_pos:   Tensor, shape [N, D]，对应的正样本第二组表征。
        temperature: float，温度参数 τ。
        
    Returns:
        loss:    scalar Tensor，InfoNCE 损失。
    """
    # 1. L2 归一化（使内积等价于余弦相似度）
    x      = F.normalize(eeg, dim=1)       # [N, D]
    x_pos  = F.normalize(eog, dim=1)   # [N, D]
    
    # 2. 计算相似度矩阵
    #    logits[i, j] = sim(x_i, x_pos_j) / τ
    logits1 = (x @ x_pos.t()) / temperature   # [N, N]
    logits2 = (x_pos @ x.t()) / temperature   # [N, N]
    
    # 3. 构造标签；正样本在对角线上
    labels = torch.arange(x.size(0), device=x.device)  # [0,1,...,N-1]
    
    # 4. 交叉熵损失
    loss1 = F.cross_entropy(logits1, labels)
    loss2 = F.cross_entropy(logits2, labels)
    return (loss1 + loss2) / 2,logits1,logits2
