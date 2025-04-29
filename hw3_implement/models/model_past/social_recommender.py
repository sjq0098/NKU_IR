import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import sys
import os
from typing import Dict, List, Tuple, Union, Optional

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class FeatureProcessor(nn.Module):
    """特征处理模块，将原始特征转换为嵌入"""
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )
        
    def forward(self, features: torch.FloatTensor) -> torch.FloatTensor:
        """前向传播"""
        # 验证输入特征
        if features is None or features.shape[0] == 0:
            raise ValueError("特征不能为空")
            
        # 如果是1D张量，扩展为2D
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
            
        return self.layers(features)


class GraphConv(nn.Module):
    """图卷积模块"""
    def __init__(self, embed_dim: int, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, user_emb: torch.Tensor, item_emb: torch.Tensor, 
                ui_adj: torch.sparse.FloatTensor, iu_adj: torch.sparse.FloatTensor, 
                social_adj: Optional[torch.sparse.FloatTensor] = None) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        前向传播
        
        Args:
            user_emb: 用户嵌入 [num_users, embed_dim]
            item_emb: 物品嵌入 [num_items, embed_dim]
            ui_adj: 用户-物品邻接矩阵
            iu_adj: 物品-用户邻接矩阵
            social_adj: 社交邻接矩阵 (可选)
            
        Returns:
            包含传播结果的字典
        """
        # 保存每层的嵌入
        ui_embeddings = [user_emb]
        social_embeddings = [user_emb] if social_adj is not None else None
        
        # 用户-物品交互传播
        u_emb, i_emb = user_emb, item_emb
        for l in range(self.n_layers):
            # 用户->物品->用户
            u_emb_temp = torch.sparse.mm(ui_adj, i_emb)
            i_emb_temp = torch.sparse.mm(iu_adj, u_emb)
            
            # 残差连接
            u_emb = u_emb + self.dropout(u_emb_temp)
            i_emb = i_emb + self.dropout(i_emb_temp)
            
            # 归一化
            u_emb = self.layer_norm(u_emb)
            i_emb = self.layer_norm(i_emb)
            
            # 保存该层嵌入
            ui_embeddings.append(u_emb)
        
        # 社交关系传播 (如果有)
        if social_adj is not None:
            u_emb_social = user_emb
            for l in range(self.n_layers):
                # 用户->用户邻居
                u_emb_temp = torch.sparse.mm(social_adj, u_emb_social)
                
                # 残差连接
                u_emb_social = u_emb_social + self.dropout(u_emb_temp)
                
                # 归一化
                u_emb_social = self.layer_norm(u_emb_social)
                
                # 保存该层嵌入
                social_embeddings.append(u_emb_social)
        
        return {
            "ui_embeddings": ui_embeddings,
            "social_embeddings": social_embeddings,
            "user_embedding": u_emb,
            "item_embedding": i_emb
        }


class AttentionLayer(nn.Module):
    """注意力层，用于聚合多层嵌入"""
    def __init__(self, embed_dim: int, num_heads: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 多头或单头注意力
        if num_heads > 1:
            self.weights = nn.Parameter(torch.randn(num_heads, embed_dim))
        else:
            self.weights = nn.Parameter(torch.randn(embed_dim))
            
        self.reset_parameters()
        
    def reset_parameters(self):
        """参数初始化"""
        if self.num_heads > 1:
            nn.init.xavier_uniform_(self.weights)
        else:
            nn.init.xavier_uniform_(self.weights.unsqueeze(0))
            
    def forward(self, layer_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            layer_embeddings: 多层嵌入列表 [n_layers+1, num_nodes, embed_dim]
            
        Returns:
            聚合后的嵌入 [num_nodes, embed_dim] 或 [num_heads, num_nodes, embed_dim]
        """
        # 检查输入
        if not layer_embeddings:
            raise ValueError("嵌入列表不能为空")
            
        # 堆叠多层嵌入
        stacked_embeddings = torch.stack(layer_embeddings, dim=0)  # [n_layers+1, num_nodes, embed_dim]
        
        # 多头注意力
        if self.num_heads > 1:
            # [num_heads, embed_dim] @ [n_layers+1, embed_dim, num_nodes] -> [num_heads, n_layers+1, num_nodes]
            attn_scores = torch.matmul(self.weights.unsqueeze(1), 
                                      stacked_embeddings.permute(0, 2, 1))
            
            # 对每个头的每一层进行softmax
            attn_scores = F.softmax(attn_scores, dim=1)  # [num_heads, n_layers+1, num_nodes]
            
            # 加权求和: [num_heads, n_layers+1, 1, num_nodes] @ [1, n_layers+1, num_nodes, embed_dim]
            output = torch.matmul(attn_scores.unsqueeze(2), 
                                 stacked_embeddings.unsqueeze(0))  # [num_heads, 1, num_nodes, embed_dim]
            
            return output.squeeze(1)  # [num_heads, num_nodes, embed_dim]
        else:
            # 单头注意力: [embed_dim] @ [n_layers+1, embed_dim, num_nodes] -> [n_layers+1, num_nodes]
            attn_scores = torch.matmul(self.weights, 
                                      stacked_embeddings.permute(0, 2, 1))
            
            # 对每一层进行softmax
            attn_scores = F.softmax(attn_scores, dim=0)  # [n_layers+1, num_nodes]
            
            # 加权求和: [num_nodes, n_layers+1] @ [n_layers+1, num_nodes, embed_dim]
            output = torch.matmul(attn_scores.permute(1, 0), 
                                 stacked_embeddings)  # [num_nodes, embed_dim]
            
            return output  # [num_nodes, embed_dim]


class DualSampling(nn.Module):
    """双重采样模块，用于增强社交关系"""
    def __init__(self, embed_dim: int, temperature: float = 0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.sampling_weights = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        """参数初始化"""
        nn.init.xavier_uniform_(self.sampling_weights)
        
    def gumbel_softmax(self, logits: torch.Tensor, tau: float = 1.0, hard: bool = False) -> torch.Tensor:
        """Gumbel-Softmax采样"""
        # 生成Gumbel噪声
        gumbels = -torch.empty_like(logits).exponential_().log()
        gumbels = (logits + gumbels) / tau
        
        # Softmax
        y_soft = F.softmax(gumbels, dim=-1)
        
        # 硬采样
        if hard:
            index = y_soft.max(dim=-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
            
        return ret
    
    def forward(self, user_emb: torch.Tensor, social_adj: Optional[torch.sparse.FloatTensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            user_emb: 用户嵌入 [batch_size, embed_dim]
            social_adj: 社交邻接矩阵 (可选)
            
        Returns:
            包含采样结果的字典
        """
        batch_size = user_emb.shape[0]
        
        # 计算用户相似度
        similarity = torch.matmul(user_emb, torch.matmul(self.sampling_weights, user_emb.t()))  # [batch_size, batch_size]
        
        # Gumbel-Softmax采样
        sampled_scores = self.gumbel_softmax(similarity, tau=self.temperature)  # [batch_size, batch_size]
        
        # 如果有显式社交关系，应用伯努利掩码
        if social_adj is not None:
            # 获取社交矩阵的稠密表示
            social_dense = social_adj.to_dense()
            
            # 随机掩码
            mask = torch.rand_like(social_dense) < 0.7  # 保留70%的显式关系
            
            # 应用掩码：保留的显式关系和新采样关系的组合
            sampled_social = torch.where(mask & (social_dense > 0), 
                                        social_dense, 
                                        sampled_scores)
        else:
            # 直接使用采样分数
            sampled_social = sampled_scores
            
        return {
            "sampled_social_relations": sampled_social
        }


class SocialRecommender(nn.Module):
    """社交推荐模型"""
    def __init__(self, 
                 num_users: int, 
                 num_items: int, 
                 embed_dim: int = 64,
                 interaction_matrix: Optional[sp.spmatrix] = None,
                 user_features: Optional[np.ndarray] = None,
                 item_features: Optional[np.ndarray] = None,
                 n_layers: int = 2,
                 dropout: float = 0.1,
                 temperature: float = 0.2):
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.temperature = temperature
        
        # 嵌入层
        if interaction_matrix is not None:
            # 创建基于SVD的初始嵌入
            self._init_svd_embedding(interaction_matrix, embed_dim)
        else:
            # 随机初始化嵌入
            self.user_embedding = nn.Parameter(torch.randn(num_users, embed_dim))
            self.item_embedding = nn.Parameter(torch.randn(num_items, embed_dim))
            nn.init.xavier_uniform_(self.user_embedding)
            nn.init.xavier_uniform_(self.item_embedding)
            
        # 特征处理
        self.use_user_features = user_features is not None
        self.use_item_features = item_features is not None
        
        if self.use_user_features:
            self.user_features = torch.FloatTensor(user_features)
            self.user_feature_processor = FeatureProcessor(user_features.shape[1], embed_dim, dropout)
            
        if self.use_item_features:
            self.item_features = torch.FloatTensor(item_features)
            self.item_feature_processor = FeatureProcessor(item_features.shape[1], embed_dim, dropout)
            
        # 图卷积模块
        self.graph_conv = GraphConv(embed_dim, n_layers, dropout)
        
        # 双重采样模块
        self.dual_sampling = DualSampling(embed_dim, temperature)
        
        # 注意力模块
        self.ui_attention = AttentionLayer(embed_dim)
        self.social_attention = AttentionLayer(embed_dim)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 损失权重
        self.reg_weight = nn.Parameter(torch.tensor(0.01))
        
    def _init_svd_embedding(self, interaction_matrix: sp.spmatrix, embed_dim: int):
        """初始化基于SVD的嵌入"""
        # 转换为scipy稀疏矩阵
        if not isinstance(interaction_matrix, sp.csr_matrix):
            interaction_matrix = interaction_matrix.tocsr()
            
        try:
            # 使用截断SVD
            from scipy.sparse.linalg import svds
            u, s, vt = svds(interaction_matrix, k=embed_dim)
            
            # 调整奇异值的影响
            s_diag = np.diag(np.sqrt(s))
            
            # 创建嵌入
            user_emb = u @ s_diag
            item_emb = vt.T @ s_diag
            
            # 转换为参数
            self.user_embedding = nn.Parameter(torch.FloatTensor(user_emb))
            self.item_embedding = nn.Parameter(torch.FloatTensor(item_emb))
        except Exception as e:
            print(f"SVD初始化失败: {e}，使用随机初始化")
            self.user_embedding = nn.Parameter(torch.randn(self.num_users, embed_dim))
            self.item_embedding = nn.Parameter(torch.randn(self.num_items, embed_dim))
            nn.init.xavier_uniform_(self.user_embedding)
            nn.init.xavier_uniform_(self.item_embedding)
    
    def forward(self, users: torch.LongTensor, pos_items: torch.LongTensor, 
                neg_items: Optional[torch.LongTensor], ui_adj_mat: Dict[str, torch.sparse.FloatTensor], 
                social_adj_mat: Optional[torch.sparse.FloatTensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            users: 用户ID [batch_size]
            pos_items: 正样本物品ID [batch_size]
            neg_items: 负样本物品ID [batch_size] (可选，预测时为None)
            ui_adj_mat: 用户-物品邻接矩阵
            social_adj_mat: 社交邻接矩阵 (可选)
            
        Returns:
            包含模型输出的字典
        """
        # 判断是训练模式还是预测模式
        is_prediction = neg_items is None
        
        # 1. 获取嵌入
        # 用户嵌入
        batch_user_emb = self.user_embedding[users]
        all_user_emb = self.user_embedding
        
        # 物品嵌入
        if is_prediction:
            # 预测模式：使用所有物品
            pos_item_emb = self.item_embedding
            neg_item_emb = None
        else:
            # 训练模式：只使用正负样本物品
            pos_item_emb = self.item_embedding[pos_items]
            neg_item_emb = self.item_embedding[neg_items]
            
        all_item_emb = self.item_embedding
        
        # 2. 添加特征嵌入
        # 用户特征
        if self.use_user_features:
            # 确保特征在正确的设备上
            if not hasattr(self, '_user_features_device') or self._user_features_device != users.device:
                self.user_features = self.user_features.to(users.device)
                self._user_features_device = users.device
                
            # 处理批次用户的特征
            user_feature_emb = self.user_feature_processor(self.user_features[users])
            batch_user_emb = batch_user_emb + user_feature_emb
            
            # 处理所有用户的特征
            all_user_feature_emb = self.user_feature_processor(self.user_features)
            all_user_emb = all_user_emb + all_user_feature_emb
            
        # 物品特征
        if self.use_item_features:
            # 确保特征在正确的设备上
            if not hasattr(self, '_item_features_device') or self._item_features_device != users.device:
                self.item_features = self.item_features.to(users.device)
                self._item_features_device = users.device
                
            # 处理物品特征
            if is_prediction:
                # 处理所有物品
                all_item_feature_emb = self.item_feature_processor(self.item_features)
                pos_item_emb = pos_item_emb + all_item_feature_emb
            else:
                # 只处理批次物品
                pos_item_feature_emb = self.item_feature_processor(self.item_features[pos_items])
                pos_item_emb = pos_item_emb + pos_item_feature_emb
                
                if neg_item_emb is not None:
                    neg_item_feature_emb = self.item_feature_processor(self.item_features[neg_items])
                    neg_item_emb = neg_item_emb + neg_item_feature_emb
                    
            # 处理所有物品的特征
            all_item_feature_emb = self.item_feature_processor(self.item_features)
            all_item_emb = all_item_emb + all_item_feature_emb
        
        # 3. 图卷积传播
        gcn_results = self.graph_conv(
            all_user_emb, 
            all_item_emb, 
            ui_adj_mat['user_item'], 
            ui_adj_mat['item_user'],
            social_adj_mat
        )
        
        ui_embeddings = gcn_results["ui_embeddings"]
        social_embeddings = gcn_results["social_embeddings"]
        
        # 4. 注意力聚合
        ui_emb = self.ui_attention(ui_embeddings)  # [num_users, embed_dim]
        
        if social_embeddings is not None:
            social_emb = self.social_attention(social_embeddings)  # [num_users, embed_dim]
        else:
            # 如果没有社交嵌入，使用用户-物品嵌入代替
            social_emb = ui_emb
        
        # 5. 提取当前批次的用户嵌入
        batch_ui_emb = ui_emb[users]
        batch_social_emb = social_emb[users]
        
        # 6. 融合嵌入
        final_user_emb = self.output_layer(
            torch.cat([batch_user_emb, batch_ui_emb, batch_social_emb], dim=1)
        )
        
        # 7. 计算得分
        if is_prediction:
            # 预测模式：计算所有物品的得分
            scores = torch.matmul(final_user_emb, pos_item_emb.t())  # [batch_size, num_items]
            return scores
        else:
            # 训练模式：只计算正负样本的得分
            pos_scores = torch.sum(final_user_emb * pos_item_emb, dim=1)  # [batch_size]
            neg_scores = torch.sum(final_user_emb * neg_item_emb, dim=1)  # [batch_size]
            
            return {
                "pos_scores": pos_scores,
                "neg_scores": neg_scores,
                "user_emb": batch_user_emb,
                "pos_item_emb": pos_item_emb,
                "neg_item_emb": neg_item_emb,
                "ui_emb": batch_ui_emb,
                "social_emb": batch_social_emb,
                "final_user_emb": final_user_emb
            }
    
    def calculate_loss(self, batch_output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算损失"""
        pos_scores = batch_output["pos_scores"]
        neg_scores = batch_output["neg_scores"]
        user_emb = batch_output["user_emb"]
        pos_item_emb = batch_output["pos_item_emb"]
        neg_item_emb = batch_output["neg_item_emb"]
        
        # BPR损失
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
        
        # L2正则化
        reg_loss = torch.norm(user_emb, p=2)**2 + torch.norm(pos_item_emb, p=2)**2 + torch.norm(neg_item_emb, p=2)**2
        reg_loss = reg_loss / user_emb.shape[0]  # 归一化
        
        # 总损失
        reg_weight = torch.clamp(self.reg_weight, min=0.0, max=0.1)  # 限制范围
        total_loss = bpr_loss + reg_weight * reg_loss
        
        return {
            "total_loss": total_loss,
            "bpr_loss": bpr_loss,
            "reg_loss": reg_loss,
            "reg_weight": reg_weight
        }
    
    def predict(self, users: torch.LongTensor, ui_adj_mat: Dict[str, torch.sparse.FloatTensor], 
               social_adj_mat: Optional[torch.sparse.FloatTensor] = None) -> torch.Tensor:
        """为用户预测所有物品的分数"""
        return self.forward(users, None, None, ui_adj_mat, social_adj_mat) 