# vib.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class VIB(nn.Module):
    """
    输入: 已经由任意 encoder 得到的特征 (B, d_x)
    输出: 预测概率 p(y|z)  (B, n_classes)
    额外返回: KL 项，用于总损失
    """

    def __init__(self,
                 d_x: int,          # encoder 输出维度
                 d_z: int,          # bottleneck 维度
                 n_classes: int,
                 beta: float = 1e-3):
        super().__init__()
        self.beta = beta

        # 将 x 映射到 μ 和 logσ²
        self.fc_mu     = nn.Linear(d_x, d_z)
        self.fc_logvar = nn.Linear(d_x, d_z)

        # 任务头：z -> logits
        self.classifier = nn.Linear(d_z, n_classes)

        # 可选：初始化
        self._init_weights()

    def _init_weights(self):
        for m in (self.fc_mu, self.fc_logvar, self.classifier):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def encode(self, x):
        """返回分布参数"""
        mu     = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """重参数化采样"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x, y=None):
        """
        x: (B, d_x)
        y: (B,)  真实标签；训练阶段需要提供
        """
        mu, logvar = self.encode(x)
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu

        logits = self.classifier(z)
        probs  = F.softmax(logits, dim=1)

        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        ce = F.cross_entropy(logits, y, reduction='none')
        loss = ce + self.beta * kl 
        #loss = ce
        return z, loss,logits



