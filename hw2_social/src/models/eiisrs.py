import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *

class EIISRS(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, social_conv_layers):
        super(EIISRS, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.social_conv_layers = social_conv_layers
        
        # 用户和物品嵌入层
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 社交图卷积层
        self.social_conv_layers = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim) 
            for _ in range(social_conv_layers)
        ])
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        
        for layer in self.social_conv_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, user_ids, item_ids, social_neighbors=None, attention_mask=None):
        """
        前向传播
        Args:
            user_ids: 用户ID
            item_ids: 物品ID
            social_neighbors: 用户的社交邻居
            attention_mask: 邻居注意力掩码
        """
        # 获取用户和物品的嵌入
        user_emb = self.user_embedding(user_ids)  # [batch_size, embedding_dim]
        item_emb = self.item_embedding(item_ids)  # [num_items, embedding_dim]
        
        # 如果有社交邻居信息，进行社交图卷积
        if social_neighbors is not None:
            # 获取邻居的嵌入
            neighbor_emb = self.user_embedding(social_neighbors)  # [batch_size, max_neighbors, embedding_dim]
            
            # 对每一层进行图卷积
            for conv in self.social_conv_layers:
                # 聚合邻居信息
                if attention_mask is not None:
                    # 使用注意力机制
                    mask = attention_mask.unsqueeze(-1)  # [batch_size, max_neighbors, 1]
                    neighbor_emb = neighbor_emb * mask
                
                # 平均池化
                neighbor_info = torch.mean(neighbor_emb, dim=1)  # [batch_size, embedding_dim]
                
                # 更新用户嵌入
                user_emb = conv(user_emb + neighbor_info)
                user_emb = F.relu(user_emb)
        
        # 计算用户和所有物品的相似度得分
        scores = torch.matmul(user_emb, item_emb.t())  # [batch_size, num_items]
        
        return scores, []  # 返回得分和空的KL损失列表（简化版本）
    
    def compute_loss(self, scores, pos_items, neg_items, kl_losses):
        # BPR损失
        pos_scores = scores[torch.arange(len(pos_items)), pos_items]
        neg_scores = scores[torch.arange(len(neg_items)), neg_items]
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # VAE损失
        vae_loss = sum(kl_losses) / len(kl_losses)
        
        # 总损失
        total_loss = bpr_loss + self.config.LAMBDA_VAE * vae_loss
        
        return total_loss 