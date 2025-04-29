import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple, Union, Optional
import math

class LightGCNLayer(nn.Module):
    """LightGCN卷积层，无参数版图卷积"""
    def __init__(self):
        super().__init__()
        
    def forward(self, user_emb, item_emb, adj_dict):
        """前向传播
        
        Args:
            user_emb: 用户嵌入 [num_users, embed_dim]
            item_emb: 物品嵌入 [num_items, embed_dim]
            adj_dict: 邻接矩阵字典，包含ui_adj和iu_adj
            
        Returns:
            更新后的用户和物品嵌入
        """
        ui_adj = adj_dict['user_item']
        iu_adj = adj_dict['item_user']
        
        # 用户->物品->用户 传播
        updated_users = torch.sparse.mm(ui_adj, item_emb)
        
        # 物品->用户->物品 传播
        updated_items = torch.sparse.mm(iu_adj, user_emb)
        
        return updated_users, updated_items


class SocialGCNLayer(nn.Module):
    """社交图卷积层，专门处理社交关系"""
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)
        self.social_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        nn.init.xavier_uniform_(self.social_weight)
        
    def forward(self, user_emb: torch.Tensor, social_adj: torch.sparse.FloatTensor) -> torch.Tensor:
        """前向传播
        
        Args:
            user_emb: 用户嵌入 [num_users, embed_dim]
            social_adj: 社交邻接矩阵
            
        Returns:
            更新后的用户嵌入
        """
        # 社交图卷积
        weighted_emb = torch.mm(user_emb, self.social_weight)  # [num_users, embed_dim]
        neighbor_emb = torch.sparse.mm(social_adj, weighted_emb)  # [num_users, embed_dim]
        
        # 残差连接
        updated_emb = user_emb + self.dropout(neighbor_emb)
        
        return updated_emb


class MultiHeadAttention(nn.Module):
    """多头自注意力层"""
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        """多头注意力前向传播
        
        Args:
            query: 查询张量 [batch_size, seq_len_q, embed_dim]
            key: 键张量 [batch_size, seq_len_k, embed_dim]
            value: 值张量 [batch_size, seq_len_v, embed_dim]
            mask: 掩码张量 (可选)
            
        Returns:
            注意力输出
        """
        batch_size = query.size(0)
        
        # 线性投影
        q = self.q_proj(query)  # [batch_size, seq_len_q, embed_dim]
        k = self.k_proj(key)  # [batch_size, seq_len_k, embed_dim]
        v = self.v_proj(value)  # [batch_size, seq_len_v, embed_dim]
        
        # 重塑为多头形式
        q = q.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len_q, head_dim]
        k = k.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len_k, head_dim]
        v = v.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len_v, head_dim]
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [batch_size, num_heads, seq_len_q, seq_len_k]
        
        # 应用掩码 (如果有)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # 注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, num_heads, seq_len_q, seq_len_k]
        attn_weights = self.dropout(attn_weights)
        
        # 注意力输出
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len_q, head_dim]
        
        # 重新整合多头输出
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.embed_dim)  # [batch_size, seq_len_q, embed_dim]
        
        # 最终线性投影
        output = self.out_proj(attn_output)  # [batch_size, seq_len_q, embed_dim]
        
        return output, attn_weights


class FeatureExtractor(nn.Module):
    """特征提取器，从原始特征中提取高级特征"""
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout: float = 0.1):
        super().__init__()
        
        all_dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList()
        
        for i in range(len(all_dims) - 1):
            self.layers.append(nn.Linear(all_dims[i], all_dims[i+1]))
            if i < len(all_dims) - 2:  # 非最后一层
                self.layers.append(nn.LayerNorm(all_dims[i+1]))
                self.layers.append(nn.LeakyReLU())
                self.layers.append(nn.Dropout(dropout))
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        for layer in self.layers:
            x = layer(x)
        return x


class SelfSupervisedModule(nn.Module):
    """自监督学习模块，通过辅助任务提升表示质量"""
    def __init__(self, embed_dim: int, temperature: float = 0.07):
        super().__init__()
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.temperature = temperature
        
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor = None):
        """前向传播
        
        Args:
            z_i: 原始嵌入
            z_j: 增强嵌入 (可选)
            
        Returns:
            对比损失
        """
        if z_j is None:
            z_j = z_i
            
        batch_size = z_i.shape[0]
        
        # 投影
        h_i = F.normalize(self.proj(z_i), dim=1)
        h_j = F.normalize(self.proj(z_j), dim=1)
        
        # 计算批内所有样本对的相似度
        sim = torch.mm(h_i, h_j.t()) / self.temperature
        
        # 对角线上的元素是正样本对
        pos_mask = torch.eye(batch_size, device=z_i.device)
        pos_sim = torch.sum(sim * pos_mask, dim=1)
        
        # 对比损失
        neg_sim = torch.logsumexp(sim, dim=1)
        loss = -torch.mean(pos_sim - neg_sim)
        
        return loss


class EnhancedSocialRecommender(nn.Module):
    """增强的社交推荐模型，结合多种优化技术"""
    def __init__(self, 
                 num_users: int, 
                 num_items: int, 
                 embed_dim: int = 64,
                 interaction_matrix: Optional[sp.spmatrix] = None,
                 social_matrix: Optional[sp.spmatrix] = None,
                 n_layers: int = 3,
                 lightgcn_layers: int = 2,
                 social_layers: int = 2,
                 num_heads: int = 4,
                 temperature: float = 0.2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.lightgcn_layers = lightgcn_layers
        self.social_layers = social_layers
        self.temperature = temperature
        
        # 初始嵌入 - 使用Xavier正态初始化，提高稳定性
        self.user_embedding = nn.Parameter(torch.randn(num_users, embed_dim) * 0.1)
        self.item_embedding = nn.Parameter(torch.randn(num_items, embed_dim) * 0.1)
        nn.init.xavier_normal_(self.user_embedding, gain=0.1)
        nn.init.xavier_normal_(self.item_embedding, gain=0.1)
        
        # LightGCN层 (无参数卷积)
        self.lightgcn_layers_list = nn.ModuleList([
            LightGCNLayer() for _ in range(lightgcn_layers)
        ])
        
        # 社交图卷积层 - 增加图卷积层的深度和宽度
        self.social_layers_list = nn.ModuleList([
            SocialGCNLayer(embed_dim, dropout) for _ in range(social_layers)
        ])
        
        # 改进型多头注意力 - 增加头数
        self.user_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.item_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.social_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # 对所有层输出加权聚合的参数
        self.layer_weights = nn.Parameter(torch.ones(n_layers + 1))
        
        # 增加跨域信息融合模块
        self.cross_domain_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        
        # 增强的输出层，使用残差连接
        self.output_user = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        self.output_item = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        
        # 自监督学习模块，提高模型泛化能力
        self.ssl_module = SelfSupervisedModule(embed_dim, temperature=0.05)  # 降低温度系数
        
        # 元路径融合模块
        self.metapath_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        
        # 高阶社交信息处理模块
        self.high_order_social = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        
        # 动态损失平衡权重 (可学习)
        self.reg_weight = nn.Parameter(torch.tensor(0.01))
        self.ssl_weight = nn.Parameter(torch.tensor(0.1))
        self.social_weight = nn.Parameter(torch.tensor(0.1))
        
    def _get_lighgcn_embeddings(self, ui_adj_mat):
        """获取LightGCN嵌入，改进了消息传递机制"""
        user_embeddings_list = [self.user_embedding]
        item_embeddings_list = [self.item_embedding]
        
        user_emb, item_emb = self.user_embedding, self.item_embedding
        
        for layer in self.lightgcn_layers_list:
            user_emb, item_emb = layer(user_emb, item_emb, ui_adj_mat)
            
            # 残差连接增强
            user_emb = user_emb + 0.1 * self.user_embedding
            item_emb = item_emb + 0.1 * self.item_embedding
            
            user_embeddings_list.append(user_emb)
            item_embeddings_list.append(item_emb)
            
        return user_embeddings_list, item_embeddings_list
    
    def _aggregate_embeddings(self, embeddings_list):
        """聚合多层嵌入，使用学习权重聚合"""
        stacked = torch.stack(embeddings_list, dim=1)  # [num_nodes, n_layers, embed_dim]
        
        # 获取层权重并归一化
        norm_weights = F.softmax(self.layer_weights[:len(embeddings_list)], dim=0)
        
        # 加权平均
        weighted_emb = torch.zeros_like(embeddings_list[0])
        for i, weight in enumerate(norm_weights):
            weighted_emb += weight * embeddings_list[i]
        
        # 获取最后一层
        last_emb = embeddings_list[-1]
        
        # 简单平均
        mean_emb = torch.mean(stacked, dim=1)
        
        # 将嵌入和聚合结果一起返回
        return stacked, weighted_emb, last_emb, mean_emb
    
    def _get_social_embeddings(self, user_emb, social_adj_mat):
        """获取社交嵌入，增强了社交信息利用"""
        social_embeddings_list = [user_emb]
        
        current_emb = user_emb
        for layer in self.social_layers_list:
            # 如果没有社交信息，跳过
            if social_adj_mat is None:
                social_embeddings_list.append(current_emb)
                continue
                
            # 应用社交图卷积
            new_emb = layer(current_emb, social_adj_mat)
            
            # 增加门控机制，自适应融合社交信息
            gate = torch.sigmoid(self.high_order_social(new_emb))
            current_emb = gate * new_emb + (1 - gate) * current_emb
            
            social_embeddings_list.append(current_emb)
            
        return social_embeddings_list
    
    def forward(self, users: torch.LongTensor, pos_items: torch.LongTensor, 
                neg_items: Optional[torch.LongTensor], ui_adj_mat: Dict[str, torch.sparse.FloatTensor], 
                social_adj_mat: Optional[torch.sparse.FloatTensor] = None) -> Dict[str, torch.Tensor]:
        """前向传播，增强了用户-物品-社交信息的交互"""
        # 判断是训练模式还是预测模式
        is_prediction = neg_items is None
        
        # Step 1: 获取LightGCN嵌入
        user_embeddings_list, item_embeddings_list = self._get_lighgcn_embeddings(ui_adj_mat)
        
        # Step 2: 聚合LightGCN嵌入
        user_stacked, user_weighted, user_last, user_mean = self._aggregate_embeddings(user_embeddings_list)
        item_stacked, item_weighted, item_last, item_mean = self._aggregate_embeddings(item_embeddings_list)
        
        # Step 3: 获取社交嵌入
        social_embeddings_list = self._get_social_embeddings(user_last, social_adj_mat)
        social_stacked, social_weighted, social_last, social_mean = self._aggregate_embeddings(social_embeddings_list)
        
        # Step 4: 融合不同类型的嵌入 (元路径融合)
        user_social_fused = self.metapath_fusion(
            torch.cat([user_weighted, social_weighted], dim=-1)
        )
        
        # Step 5: 使用多头注意力聚合嵌入
        # 准备注意力输入
        batch_size = 1 if len(users.shape) == 0 else users.shape[0]
        
        # 提取当前批次的用户嵌入
        if batch_size > 1:
            batch_user_stacked = user_stacked[users]   # [batch_size, n_layers, embed_dim]
            batch_social_stacked = social_stacked[users]   # [batch_size, n_layers, embed_dim]
        else:
            batch_user_stacked = user_stacked[users].unsqueeze(0)  # [1, n_layers, embed_dim]
            batch_social_stacked = social_stacked[users].unsqueeze(0)  # [1, n_layers, embed_dim]
        
        # Step 6: 增强注意力融合
        # 应用多头注意力
        user_attn_out, user_attn_weights = self.user_attention(
            batch_user_stacked, batch_user_stacked, batch_user_stacked
        )  # [batch_size, n_layers, embed_dim]
        
        social_attn_out, social_attn_weights = self.social_attention(
            batch_social_stacked, batch_social_stacked, batch_social_stacked
        )  # [batch_size, n_layers, embed_dim]
        
        # 跨域注意力: 让社交信息关注用户表示，增强相关性
        cross_attn_out, _ = self.social_attention(
            batch_user_stacked, batch_social_stacked, batch_social_stacked
        )  # [batch_size, n_layers, embed_dim]
        
        # 取最后一层作为注意力输出
        user_attn_emb = user_attn_out[:, -1, :]  # [batch_size, embed_dim]
        social_attn_emb = social_attn_out[:, -1, :]  # [batch_size, embed_dim]
        cross_attn_emb = cross_attn_out[:, -1, :]  # [batch_size, embed_dim]
        
        # 跨域融合 (用户-社交)
        cross_fusion = self.cross_domain_fusion(
            torch.cat([user_attn_emb, social_attn_emb], dim=-1)
        )
        
        # Step 7: 整合所有嵌入
        batch_user_weighted = user_weighted[users]  # [batch_size, embed_dim]
        batch_social_weighted = social_weighted[users]  # [batch_size, embed_dim]
        
        # 融合用户表示 (增强版)
        final_user_emb = self.output_user(
            torch.cat([user_attn_emb, cross_fusion, social_attn_emb], dim=-1)
        )  # [batch_size, embed_dim]
        
        # 增加残差连接
        final_user_emb = final_user_emb + batch_user_weighted * 0.2 + batch_social_weighted * 0.1
        
        # Step 8: 处理物品嵌入
        if is_prediction:
            # 预测模式: 计算所有物品的得分
            # 使用注意力和批处理处理所有物品的嵌入
            _, all_item_attn_out, _ = self._batch_attention(
                self.item_attention, item_stacked, max_batch_size=512
            )
            
            # 融合物品表示
            all_item_weighted = item_weighted  # [num_items, embed_dim]
            
            # 强化物品表示
            all_item_emb = self.output_item(
                torch.cat([all_item_attn_out, all_item_weighted], dim=-1)
            )  # [num_items, embed_dim]
            
            # 残差连接
            all_item_emb = all_item_emb + all_item_weighted * 0.2
            
            # 计算得分
            scores = torch.matmul(final_user_emb, all_item_emb.t())  # [batch_size, num_items]
            scores = scores / self.temperature  # 应用温度系数
            
            return {
                "scores": scores,
                "user_embeddings": final_user_emb,
                "item_embeddings": all_item_emb
            }
        else:
            # 训练模式: 只处理正负样本物品
            pos_item_stacked = item_stacked[pos_items]  # [batch_size, n_layers, embed_dim]
            pos_item_weighted = item_weighted[pos_items]  # [batch_size, embed_dim]
            
            pos_item_attn_out, _ = self.item_attention(
                pos_item_stacked, pos_item_stacked, pos_item_stacked
            )  # [batch_size, n_layers, embed_dim]
            pos_item_attn_emb = pos_item_attn_out[:, -1, :]  # [batch_size, embed_dim]
            
            # 融合正样本物品表示
            final_pos_item_emb = self.output_item(
                torch.cat([pos_item_attn_emb, pos_item_weighted], dim=-1)
            )  # [batch_size, embed_dim]
            
            # 残差连接
            final_pos_item_emb = final_pos_item_emb + pos_item_weighted * 0.2
            
            # 处理负样本
            if neg_items.dim() > 1:
                # 多个负样本
                batch_size, num_negs = neg_items.shape
                flat_neg_items = neg_items.reshape(-1)  # [batch_size * num_negs]
                
                # 获取负样本的嵌入
                flat_neg_item_stacked = item_stacked[flat_neg_items]  # [batch_size * num_negs, n_layers, embed_dim]
                flat_neg_item_weighted = item_weighted[flat_neg_items]  # [batch_size * num_negs, embed_dim]
                
                # 应用注意力
                flat_neg_item_attn_out, _ = self.item_attention(
                    flat_neg_item_stacked, flat_neg_item_stacked, flat_neg_item_stacked
                )  # [batch_size * num_negs, n_layers, embed_dim]
                flat_neg_item_attn_emb = flat_neg_item_attn_out[:, -1, :]  # [batch_size * num_negs, embed_dim]
                
                # 融合负样本物品表示
                flat_final_neg_item_emb = self.output_item(
                    torch.cat([flat_neg_item_attn_emb, flat_neg_item_weighted], dim=-1)
                )  # [batch_size * num_negs, embed_dim]
                
                # 残差连接
                flat_final_neg_item_emb = flat_final_neg_item_emb + flat_neg_item_weighted * 0.2
                
                # 重塑为 [batch_size, num_negs, embed_dim]
                final_neg_item_emb = flat_final_neg_item_emb.reshape(batch_size, num_negs, -1)
                
                # 计算得分 - 增强的评分机制
                pos_scores = torch.sum(final_user_emb.unsqueeze(1) * final_pos_item_emb.unsqueeze(1), dim=-1).squeeze(1)  # [batch_size]
                neg_scores = torch.sum(final_user_emb.unsqueeze(1) * final_neg_item_emb, dim=-1)  # [batch_size, num_negs]
            else:
                # 单个负样本
                neg_item_stacked = item_stacked[neg_items]  # [batch_size, n_layers, embed_dim]
                neg_item_weighted = item_weighted[neg_items]  # [batch_size, embed_dim]
                
                neg_item_attn_out, _ = self.item_attention(
                    neg_item_stacked, neg_item_stacked, neg_item_stacked
                )  # [batch_size, n_layers, embed_dim]
                neg_item_attn_emb = neg_item_attn_out[:, -1, :]  # [batch_size, embed_dim]
                
                # 融合负样本物品表示
                final_neg_item_emb = self.output_item(
                    torch.cat([neg_item_attn_emb, neg_item_weighted], dim=-1)
                )  # [batch_size, embed_dim]
                
                # 残差连接
                final_neg_item_emb = final_neg_item_emb + neg_item_weighted * 0.2
                
                # 计算得分
                pos_scores = torch.sum(final_user_emb * final_pos_item_emb, dim=-1)  # [batch_size]
                neg_scores = torch.sum(final_user_emb * final_neg_item_emb, dim=-1)  # [batch_size]
            
            # 使用温度系数
            pos_scores = pos_scores / self.temperature
            neg_scores = neg_scores / self.temperature
            
            # 自监督学习损失 - 改进的对比学习
            ssl_loss_ui = self.ssl_module(user_attn_emb, pos_item_attn_emb)  # 用户-物品对比
            ssl_loss_us = self.ssl_module(user_attn_emb, social_attn_emb)  # 用户-社交对比
            ssl_loss = 0.5 * ssl_loss_ui + 0.5 * ssl_loss_us  # 组合损失
            
            return {
                "pos_scores": pos_scores,
                "neg_scores": neg_scores,
                "ssl_loss": ssl_loss,
                "user_emb": final_user_emb,
                "pos_item_emb": final_pos_item_emb,
                "neg_item_emb": final_neg_item_emb,
                "user_attn_emb": user_attn_emb,
                "social_attn_emb": social_attn_emb
            }
    
    def _batch_attention(self, attention_layer, stacked_emb, max_batch_size=512):
        """分批处理大型注意力计算，节省内存"""
        num_nodes = stacked_emb.shape[0]
        num_batches = (num_nodes + max_batch_size - 1) // max_batch_size
        
        outputs = []
        attentions = []
        
        for i in range(num_batches):
            start_idx = i * max_batch_size
            end_idx = min((i + 1) * max_batch_size, num_nodes)
            batch = stacked_emb[start_idx:end_idx]  # [batch_size, n_layers, embed_dim]
            
            output, attn = attention_layer(batch, batch, batch)  # [batch_size, n_layers, embed_dim]
            outputs.append(output)
            attentions.append(attn)
        
        # 合并所有批次的结果
        all_outputs = torch.cat(outputs, dim=0)  # [num_nodes, n_layers, embed_dim]
        all_attentions = torch.cat(attentions, dim=0)  # [num_nodes, num_heads, n_layers, n_layers]
        
        # 提取最后一层的输出
        last_layer_output = all_outputs[:, -1, :]  # [num_nodes, embed_dim]
        
        return all_outputs, last_layer_output, all_attentions
    
    def calculate_loss(self, batch_output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算综合损失 - 改进了损失函数组合策略"""
        # 预测模式直接返回得分
        if "scores" in batch_output:
            return {"scores": batch_output["scores"]}
            
        # 提取张量
        pos_scores = batch_output["pos_scores"]
        neg_scores = batch_output["neg_scores"]
        ssl_loss = batch_output["ssl_loss"]
        user_emb = batch_output["user_emb"]
        pos_item_emb = batch_output["pos_item_emb"]
        
        # 处理多负样本情况
        if neg_scores.dim() > 1:
            # 对每个正样本，使用所有负样本计算BPR损失
            batch_size, num_negs = neg_scores.shape
            
            # 将正样本分数重复num_negs次
            pos_scores_expanded = pos_scores.unsqueeze(1).expand(batch_size, num_negs).reshape(-1)
            neg_scores_flat = neg_scores.reshape(-1)
            
            # 改进的BPR损失 - 使用目标边缘
            margin = 1.0
            bpr_loss = torch.mean(torch.relu(margin - (pos_scores_expanded - neg_scores_flat)))
            
            # 对负样本嵌入求平均，用于正则化
            if batch_output["neg_item_emb"].dim() > 2:
                neg_item_emb_avg = torch.mean(batch_output["neg_item_emb"], dim=1)
            else:
                neg_item_emb_avg = batch_output["neg_item_emb"]
        else:
            # 单负样本情况 - 标准BPR损失
            # bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
            # 使用更稳定的带边缘的损失
            margin = 1.0
            bpr_loss = torch.mean(torch.relu(margin - (pos_scores - neg_scores)))
            neg_item_emb_avg = batch_output["neg_item_emb"]
        
        # 改进的正则化 - 更强调嵌入的紧凑性
        reg_loss = (torch.norm(user_emb, p=2)**2 + 
                   torch.norm(pos_item_emb, p=2)**2 + 
                   torch.norm(neg_item_emb_avg, p=2)**2)
        reg_loss = reg_loss / user_emb.shape[0]  # 归一化
        
        # 社交损失 - 鼓励用户表示和社交表示相似性
        if "user_attn_emb" in batch_output and "social_attn_emb" in batch_output:
            user_attn_emb = batch_output["user_attn_emb"]
            social_attn_emb = batch_output["social_attn_emb"]
            
            # 使用余弦相似度而非欧氏距离
            user_emb_norm = F.normalize(user_attn_emb, p=2, dim=1)
            social_emb_norm = F.normalize(social_attn_emb, p=2, dim=1)
            social_loss = 1 - torch.mean(torch.sum(user_emb_norm * social_emb_norm, dim=1))
        else:
            social_loss = torch.tensor(0.0, device=user_emb.device)
        
        # 动态调整权重
        reg_weight = torch.clamp(self.reg_weight, min=0.001, max=0.05)  # 减小正则化影响
        ssl_weight = torch.clamp(self.ssl_weight, min=0.05, max=0.5)   # 增强自监督学习
        social_weight = torch.clamp(self.social_weight, min=0.05, max=0.5)  # 增强社交约束
        
        # 计算总损失 - 平衡各成分
        total_loss = bpr_loss + reg_weight * reg_loss + ssl_weight * ssl_loss + social_weight * social_loss
        
        return {
            "total_loss": total_loss,
            "bpr_loss": bpr_loss,
            "reg_loss": reg_loss,
            "ssl_loss": ssl_loss,
            "social_loss": social_loss,
            "reg_weight": reg_weight,
            "ssl_weight": ssl_weight,
            "social_weight": social_weight
        } 