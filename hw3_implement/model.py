import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import convert_sparse_matrix_to_sparse_tensor

class LightGCNLayer(nn.Module):
    """LightGCN图卷积层 (无权重参数)"""
    def __init__(self):
        super(LightGCNLayer, self).__init__()
    
    def forward(self, x, adj):
        """前向传播: H_next = A * H"""
        return torch.sparse.mm(adj, x)

class SocialRecommender(nn.Module):
    """社交推荐模型 (简化版)"""
    def __init__(self, 
                 num_users: int, 
                 num_items: int, 
                 embed_dim: int = 64,
                 n_layers: int = 3,
                 social_layers: int = 2,
                 dropout: float = 0.1,
                 temperature: float = 0.1,
                 device: str = 'cuda'):
        """初始化模型"""
        super(SocialRecommender, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers # 用户-物品图层数
        self.social_layers = social_layers # 社交图层数
        self.dropout = dropout
        self.temperature = temperature
        self.device = device
        
        # 嵌入层
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        
        # 初始化嵌入权重
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # 特征融合层 (如果使用SVD)
        # 假设SVD特征维度与embed_dim相同
        self.user_fusion = nn.Linear(embed_dim * 2, embed_dim)
        self.item_fusion = nn.Linear(embed_dim * 2, embed_dim)
            
        # Dropout层
        self.dropout_layer = nn.Dropout(dropout)
        
        # 预计算的矩阵 (在main脚本中设置)
        self.R_norm_sp = None
        self.R_item_norm_sp = None
        self.S_norm_sp = None
        self.user_svd = None
        self.item_svd = None

    def forward(self):
        """执行图卷积传播，返回最终的用户和物品嵌入"""
        
        # 1. 初始嵌入 (融合SVD，如果提供)
        user_emb_0 = self.user_embedding.weight
        item_emb_0 = self.item_embedding.weight
        
        if self.user_svd is not None and self.item_svd is not None:
            user_feature_emb = torch.cat([user_emb_0, self.user_svd], dim=1)
            item_feature_emb = torch.cat([item_emb_0, self.item_svd], dim=1)
            user_emb_0 = self.user_fusion(user_feature_emb)
            item_emb_0 = self.item_fusion(item_feature_emb)
        
        # 存储各层嵌入 (用于最终聚合)
        all_user_embeds = [user_emb_0]
        all_item_embeds = [item_emb_0]

        # 2. 用户-物品图传播 (LightGCN风格)
        current_user_embeds = user_emb_0
        current_item_embeds = item_emb_0
        
        for _ in range(self.n_layers):
            # 更新用户嵌入: 聚合连接的物品信息
            # R_norm_sp: [num_users, num_items], current_item_embeds: [num_items, embed_dim]
            next_user_embeds = torch.sparse.mm(self.R_norm_sp, current_item_embeds)
            
            # 更新物品嵌入: 聚合连接的用户信息
            # R_item_norm_sp: [num_items, num_users], current_user_embeds: [num_users, embed_dim]
            next_item_embeds = torch.sparse.mm(self.R_item_norm_sp, current_user_embeds)
            
            # 更新嵌入 (这里可以考虑是否添加残差连接，暂时不加)
            current_user_embeds = next_user_embeds
            current_item_embeds = next_item_embeds
            
            # 应用Dropout
            current_user_embeds = self.dropout_layer(current_user_embeds)
            current_item_embeds = self.dropout_layer(current_item_embeds)
            
            # 保存该层嵌入
            all_user_embeds.append(current_user_embeds)
            all_item_embeds.append(current_item_embeds)

        # 3. 社交图传播 (如果存在)
        social_user_embeds = user_emb_0 # 初始社交嵌入
        all_social_user_embeds = [social_user_embeds]
        
        if self.S_norm_sp is not None:
            current_social_user_embeds = social_user_embeds
            for _ in range(self.social_layers):
                # 社交邻居信息聚合
                next_social_user_embeds = torch.sparse.mm(self.S_norm_sp, current_social_user_embeds)
                
                # 更新嵌入
                current_social_user_embeds = next_social_user_embeds
                current_social_user_embeds = self.dropout_layer(current_social_user_embeds)
                all_social_user_embeds.append(current_social_user_embeds)

        # 4. 聚合各层嵌入
        # 用户-物品图聚合
        user_embeds_final = torch.stack(all_user_embeds, dim=1)
        user_embeds_final = torch.mean(user_embeds_final, dim=1)
        
        item_embeds_final = torch.stack(all_item_embeds, dim=1)
        item_embeds_final = torch.mean(item_embeds_final, dim=1)
        
        # 社交图聚合
        social_user_embeds_final = torch.stack(all_social_user_embeds, dim=1)
        social_user_embeds_final = torch.mean(social_user_embeds_final, dim=1)

        # 5. 融合用户-物品嵌入和社交嵌入 (简单加权)
        final_user_representation = 0.6 * user_embeds_final + 0.4 * social_user_embeds_final
        final_item_representation = item_embeds_final

        # 归一化最终嵌入
        final_user_representation = F.normalize(final_user_representation, p=2, dim=1)
        final_item_representation = F.normalize(final_item_representation, p=2, dim=1)
        
        return final_user_representation, final_item_representation

    def predict(self, user_embeds, item_embeds, users):
        """预测用户对所有物品的评分"""
        u_embeds = user_embeds[users]
        scores = torch.matmul(u_embeds, item_embeds.t())
        return scores

    def bpr_loss(self, users, pos_items, neg_items, user_embeds, item_embeds):
        """计算BPR损失"""
        user_e = user_embeds[users]
        pos_e = item_embeds[pos_items]
        neg_e = item_embeds[neg_items]
        
        pos_scores = torch.sum(user_e * pos_e, dim=1)
        neg_scores = torch.sum(user_e * neg_e, dim=1)
        
        loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # 添加L2正则化
        reg_loss = 1e-4 * (torch.norm(user_e)**2 + torch.norm(pos_e)**2 + torch.norm(neg_e)**2) / len(users)
        
        return loss + reg_loss

    def infonce_loss(self, users, pos_items, user_embeds, item_embeds):
        """计算InfoNCE损失"""
        user_e = F.normalize(user_embeds[users], dim=1)
        pos_e = F.normalize(item_embeds[pos_items], dim=1)
        all_item_e = F.normalize(item_embeds, dim=1)
        
        pos_scores = torch.sum(user_e * pos_e, dim=1) / self.temperature
        all_scores = torch.matmul(user_e, all_item_e.t()) / self.temperature
        
        # 防止计算log(0)
        exp_all_scores = torch.exp(all_scores)
        log_sum_exp = torch.log(torch.sum(exp_all_scores, dim=1) + 1e-10)
        
        loss = -torch.mean(pos_scores - log_sum_exp)
        
        # 添加正则化
        reg_loss = 1e-5 * (torch.norm(user_e)**2 + torch.norm(pos_e)**2) / len(users)
        
        return loss + reg_loss 