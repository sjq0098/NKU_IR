import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfGatingUnit(nn.Module):
    """自门控单元，用于平衡不同路径的信息"""
    def __init__(self, dim):
        super(SelfGatingUnit, self).__init__()
        self.gate = nn.Linear(dim, dim)
        
    def forward(self, x):
        gate = torch.sigmoid(self.gate(x))
        return x * gate

class LightGCNLayer(nn.Module):
    """轻量级图卷积层"""
    def __init__(self, dim, dropout=0.1):
        super(LightGCNLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj):
        x = self.dropout(x)
        return torch.matmul(adj, x)

class LGEVAE(nn.Module):
    """层级图增强变分自编码器"""
    def __init__(self, dim, num_layers):
        super(LGEVAE, self).__init__()
        self.num_layers = num_layers
        
        # 编码器
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim * 2)
            ) for _ in range(num_layers)
        ])
        
        # 解码器
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim)
            ) for _ in range(num_layers)
        ])
        
    def encode(self, x, layer_idx):
        mu_logvar = self.encoders[layer_idx](x)
        mu, logvar = torch.chunk(mu_logvar, 2, dim=-1)
        return mu, logvar
    
    def decode(self, z, layer_idx):
        return self.decoders[layer_idx](z)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class HierarchicalAttention(nn.Module):
    """层级图注意力网络"""
    def __init__(self, dim):
        super(HierarchicalAttention, self).__init__()
        self.attention = nn.Linear(dim, 1)
        
    def forward(self, x):
        # x shape: [num_layers, batch_size, dim]
        attn = F.softmax(self.attention(x), dim=0)
        return torch.sum(x * attn, dim=0)

class GumbelSampler(nn.Module):
    """Gumbel采样器，用于选择隐式邻居"""
    def __init__(self, temperature=0.1):
        super(GumbelSampler, self).__init__()
        self.temperature = temperature
        
    def forward(self, logits):
        # 添加Gumbel噪声
        noise = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(noise + 1e-10) + 1e-10)
        return F.softmax((logits + gumbel_noise) / self.temperature, dim=-1) 