import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple, Union, Optional
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

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
        
        # 添加图卷积参数
        self.W_gc = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(n_layers)
        ])
        
    def forward(self, user_emb: torch.Tensor, item_emb: torch.Tensor, 
                ui_adj: torch.sparse.FloatTensor, iu_adj: torch.sparse.FloatTensor, 
                social_adj: Optional[torch.sparse.FloatTensor] = None) -> Dict:
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
            u_emb_temp = self.W_gc[l](u_emb_temp)  # 应用图卷积权重
            
            i_emb_temp = torch.sparse.mm(iu_adj, u_emb)
            i_emb_temp = self.W_gc[l](i_emb_temp)  # 应用图卷积权重
            
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
                u_emb_temp = self.W_gc[l](u_emb_temp)  # 应用图卷积权重
                
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


class EnhancedAttentionLayer(nn.Module):
    """增强的注意力层，用于聚合多层嵌入"""
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 多头注意力参数
        self.Q = nn.Linear(embed_dim, embed_dim * num_heads)
        self.K = nn.Linear(embed_dim, embed_dim * num_heads)
        self.V = nn.Linear(embed_dim, embed_dim * num_heads)
        self.O = nn.Linear(embed_dim * num_heads, embed_dim)
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.reset_parameters()
        
    def reset_parameters(self):
        """参数初始化"""
        nn.init.xavier_uniform_(self.Q.weight)
        nn.init.xavier_uniform_(self.K.weight)
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.O.weight)
            
    def forward(self, layer_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            layer_embeddings: 多层嵌入列表 [n_layers+1, num_nodes, embed_dim]
            
        Returns:
            聚合后的嵌入 [num_nodes, embed_dim]
        """
        # 检查输入
        if not layer_embeddings:
            raise ValueError("嵌入列表不能为空")
        
        # 如果只有一层，直接返回
        if len(layer_embeddings) == 1:
            return layer_embeddings[0]
            
        try:
            # 叠加所有层的嵌入，得到 [n_layers+1, num_nodes, embed_dim]
            stacked_embeddings = torch.stack(layer_embeddings, dim=0)
            
            # 转置为 [num_nodes, n_layers+1, embed_dim]
            transposed_embeddings = stacked_embeddings.transpose(0, 1)
            batch_size, seq_len, _ = transposed_embeddings.shape
            
            # 多头注意力计算
            q = self.Q(transposed_embeddings).view(batch_size, seq_len, self.num_heads, self.embed_dim).transpose(1, 2)
            k = self.K(transposed_embeddings).view(batch_size, seq_len, self.num_heads, self.embed_dim).transpose(1, 2)
            v = self.V(transposed_embeddings).view(batch_size, seq_len, self.num_heads, self.embed_dim).transpose(1, 2)
            
            # 计算注意力分数
            attention_scores = torch.matmul(q, k.transpose(2, 3)) / (self.embed_dim ** 0.5)
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            # 加权聚合
            context = torch.matmul(attention_weights, v)
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.embed_dim)
            output = self.O(context)
            
            # 由于我们只需要每个节点的单一嵌入，取最后一层的结果
            aggregated_emb = output[:, -1, :]
            
            return self.layer_norm(aggregated_emb)
            
        except Exception as e:
            print(f"注意力聚合失败，使用简单平均: {e}")
            # 如果注意力计算失败，使用简单平均作为后备方案
            return torch.mean(torch.stack(layer_embeddings), dim=0)


class SVDFeatureExtractor:
    """SVD特征提取器，从交互矩阵中提取SVD特征"""
    def __init__(self, interaction_matrix: sp.spmatrix, num_factors: int = 16):
        self.num_factors = num_factors
        self.user_features, self.item_features = self.extract_svd_features(interaction_matrix)
        
    def extract_svd_features(self, interaction_matrix: sp.spmatrix) -> Tuple[np.ndarray, np.ndarray]:
        """使用SVD提取特征"""
        # 确保矩阵是CSR格式
        if not isinstance(interaction_matrix, sp.csr_matrix):
            interaction_matrix = interaction_matrix.tocsr()
            
        try:
            # 尝试执行SVD
            print(f"执行SVD分解，factors={self.num_factors}...")
            U, sigma, Vt = svds(interaction_matrix, k=min(self.num_factors, min(interaction_matrix.shape)-1))
            
            # 构建特征
            sigma_diag = np.diag(sigma)
            user_features = U @ sigma_diag
            item_features = Vt.T
            
            # 检查是否需要填充到目标维度
            if user_features.shape[1] < self.num_factors:
                print(f"SVD特征维度({user_features.shape[1]})不足要求({self.num_factors})，进行填充...")
                padding = np.zeros((user_features.shape[0], self.num_factors - user_features.shape[1]))
                user_features = np.hstack([user_features, padding])
                
                padding = np.zeros((item_features.shape[0], self.num_factors - item_features.shape[1]))
                item_features = np.hstack([item_features, padding])
            
            # 使用copy()方法确保返回的数组没有负步长(negative strides)
            return user_features.copy(), item_features.copy()
            
        except Exception as e:
            print(f"SVD分解失败: {e}")
            print("生成随机特征作为替代...")
            # 生成随机特征作为替代
            user_features = np.random.randn(interaction_matrix.shape[0], self.num_factors) * 0.01
            item_features = np.random.randn(interaction_matrix.shape[1], self.num_factors) * 0.01
            return user_features, item_features


class CFFeatureExtractor:
    """协同过滤特征提取器，计算用户和物品的协同过滤特征"""
    def __init__(self, interaction_matrix: sp.spmatrix):
        self.user_features, self.item_features = self.extract_cf_features(interaction_matrix)
        
    def extract_cf_features(self, interaction_matrix: sp.spmatrix) -> Tuple[np.ndarray, np.ndarray]:
        """提取协同过滤特征"""
        # 确保矩阵是CSR格式
        if not isinstance(interaction_matrix, sp.csr_matrix):
            interaction_matrix = interaction_matrix.tocsr()
            
        # 用户相似度 (用户-用户协同过滤)
        user_sim_matrix = cosine_similarity(interaction_matrix)
        
        # 物品相似度 (物品-物品协同过滤)
        item_sim_matrix = cosine_similarity(interaction_matrix.T)
        
        # 提取特征：每个用户/物品的平均相似度和最大相似度
        user_avg_sim = np.mean(user_sim_matrix, axis=1, keepdims=True)
        user_max_sim = np.max(user_sim_matrix, axis=1, keepdims=True)
        user_cf_features = np.hstack([user_avg_sim, user_max_sim])
        
        item_avg_sim = np.mean(item_sim_matrix, axis=1, keepdims=True)
        item_max_sim = np.max(item_sim_matrix, axis=1, keepdims=True)
        item_cf_features = np.hstack([item_avg_sim, item_max_sim])
        
        # 使用copy()确保数组连续
        return user_cf_features.copy(), item_cf_features.copy()


class StatisticsFeatureExtractor:
    """统计特征提取器，提取交互统计特征"""
    def __init__(self, interaction_matrix: sp.spmatrix, social_matrix: Optional[sp.spmatrix] = None):
        self.user_features, self.item_features = self.extract_statistics_features(
            interaction_matrix, social_matrix
        )
        
    def extract_statistics_features(self, interaction_matrix: sp.spmatrix, 
                                  social_matrix: Optional[sp.spmatrix] = None) -> Tuple[np.ndarray, np.ndarray]:
        """提取统计特征"""
        # 确保矩阵是CSR格式
        if not isinstance(interaction_matrix, sp.csr_matrix):
            interaction_matrix = interaction_matrix.tocsr()
            
        # 用户交互次数
        user_interaction_count = np.array(interaction_matrix.sum(axis=1)).flatten()
        
        # 物品被交互次数
        item_interaction_count = np.array(interaction_matrix.sum(axis=0)).flatten()
        
        # 用户平均交互强度 (如果有权重)
        user_avg_weight = np.zeros_like(user_interaction_count)
        for i in range(interaction_matrix.shape[0]):
            row = interaction_matrix.getrow(i)
            if row.nnz > 0:
                user_avg_weight[i] = row.sum() / row.nnz
                
        # 物品平均交互强度 (如果有权重)
        item_avg_weight = np.zeros_like(item_interaction_count)
        for i in range(interaction_matrix.shape[1]):
            col = interaction_matrix.getcol(i)
            if col.nnz > 0:
                item_avg_weight[i] = col.sum() / col.nnz
        
        # 社交特征 (如果有社交矩阵)
        if social_matrix is not None:
            if not isinstance(social_matrix, sp.csr_matrix):
                social_matrix = social_matrix.tocsr()
                
            # 用户社交连接数
            user_social_count = np.array(social_matrix.sum(axis=1)).flatten()
            
            # 将所有特征组合
            user_features = np.vstack([
                user_interaction_count,
                user_avg_weight,
                user_social_count
            ]).T
        else:
            # 没有社交特征时
            user_features = np.vstack([
                user_interaction_count,
                user_avg_weight
            ]).T
            
        # 物品特征
        item_features = np.vstack([
            item_interaction_count,
            item_avg_weight
        ]).T
        
        # 归一化特征
        user_features = self._normalize_features(user_features)
        item_features = self._normalize_features(item_features)
        
        # 使用copy()确保数组连续
        return user_features.copy(), item_features.copy()
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """归一化特征"""
        # 避免除以零
        epsilon = 1e-10
        features_copy = features.copy()  # 创建副本避免原地修改
        for col in range(features_copy.shape[1]):
            min_val = np.min(features_copy[:, col])
            max_val = np.max(features_copy[:, col])
            
            if max_val - min_val > epsilon:
                features_copy[:, col] = (features_copy[:, col] - min_val) / (max_val - min_val)
            else:
                features_copy[:, col] = 0.0
                
        return features_copy


class UserTower(nn.Module):
    """用户塔架"""
    def __init__(self, input_dim: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # 多层感知机
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )
    
    def forward(self, user_emb: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.mlp(user_emb)


class ItemTower(nn.Module):
    """物品塔架"""
    def __init__(self, input_dim: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # 多层感知机
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )
    
    def forward(self, item_emb: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.mlp(item_emb)


class SocialRecommenderSimplified(nn.Module):
    """结合传统方法和深度学习的社交推荐模型"""
    def __init__(self, 
                 num_users: int, 
                 num_items: int, 
                 embed_dim: int = 64,
                 interaction_matrix: Optional[sp.spmatrix] = None,
                 social_matrix: Optional[sp.spmatrix] = None,
                 user_features: Optional[np.ndarray] = None,
                 item_features: Optional[np.ndarray] = None,
                 n_layers: int = 2,
                 dropout: float = 0.1,
                 temperature: float = 0.2):
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.use_traditional_features = False
        
        # 嵌入层
        self.user_embedding = nn.Parameter(torch.randn(num_users, embed_dim))
        self.item_embedding = nn.Parameter(torch.randn(num_items, embed_dim))
        nn.init.xavier_uniform_(self.user_embedding)
        nn.init.xavier_uniform_(self.item_embedding)
        
        # 初始化双塔输入维度默认值
        self.user_tower_input_dim = embed_dim
        self.item_tower_input_dim = embed_dim
        
        # 提取传统算法特征
        if interaction_matrix is not None:
            print(f"开始提取SVD特征...")
            try:
                # SVD特征
                svd_extractor = SVDFeatureExtractor(interaction_matrix, num_factors=embed_dim)  # 确保SVD特征维度与embed_dim一致
                self.svd_user_features = torch.FloatTensor(svd_extractor.user_features)
                self.svd_item_features = torch.FloatTensor(svd_extractor.item_features)
                print(f"SVD特征提取成功! 用户特征: {self.svd_user_features.shape}, 物品特征: {self.svd_item_features.shape}")
                
                # 协同过滤特征
                print(f"开始提取协同过滤特征...")
                cf_extractor = CFFeatureExtractor(interaction_matrix)
                self.cf_user_features = torch.FloatTensor(cf_extractor.user_features)
                self.cf_item_features = torch.FloatTensor(cf_extractor.item_features)
                print(f"协同过滤特征提取成功! 用户特征: {self.cf_user_features.shape}, 物品特征: {self.cf_item_features.shape}")
                
                # 统计特征
                print(f"开始提取统计特征...")
                stats_extractor = StatisticsFeatureExtractor(interaction_matrix, social_matrix)
                self.stats_user_features = torch.FloatTensor(stats_extractor.user_features)
                self.stats_item_features = torch.FloatTensor(stats_extractor.item_features)
                print(f"统计特征提取成功! 用户特征: {self.stats_user_features.shape}, 物品特征: {self.stats_item_features.shape}")
                
                # 特征处理器 - 确保输出维度一致
                self.svd_processor = FeatureProcessor(self.svd_user_features.shape[1], embed_dim, dropout)
                self.cf_processor = FeatureProcessor(self.cf_user_features.shape[1], embed_dim, dropout)
                self.stats_processor = FeatureProcessor(self.stats_user_features.shape[1], embed_dim, dropout)
                
                # 计算双塔输入维度
                self.user_tower_input_dim = embed_dim * 4  # 基础嵌入+SVD+CF+stats
                self.item_tower_input_dim = embed_dim * 4
                self.use_traditional_features = True
                
            except Exception as e:
                print(f"特征提取过程中发生错误: {e}")
                print("将使用简化版模型，不包含传统算法特征")
                # 如果传统特征提取失败，用默认维度
                self.user_tower_input_dim = embed_dim
                self.item_tower_input_dim = embed_dim
                self.use_traditional_features = False
        
        # 外部特征处理
        self.use_user_features = user_features is not None
        self.use_item_features = item_features is not None
        
        if self.use_user_features:
            self.ext_user_features = torch.FloatTensor(user_features)
            self.ext_user_processor = FeatureProcessor(user_features.shape[1], embed_dim, dropout)
            self.user_tower_input_dim += embed_dim
            
        if self.use_item_features:
            self.ext_item_features = torch.FloatTensor(item_features)
            self.ext_item_processor = FeatureProcessor(item_features.shape[1], embed_dim, dropout)
            self.item_tower_input_dim += embed_dim
            
        # 图卷积模块
        self.graph_conv = GraphConv(embed_dim, n_layers, dropout)
        
        # 双塔结构 - 用正确计算的输入维度
        print(f"用户塔输入维度: {self.user_tower_input_dim}, 物品塔输入维度: {self.item_tower_input_dim}")
        self.user_tower = UserTower(self.user_tower_input_dim, embed_dim, dropout)
        self.item_tower = ItemTower(self.item_tower_input_dim, embed_dim, dropout)
        
        # 增强的注意力模块
        self.ui_attention = EnhancedAttentionLayer(embed_dim, num_heads=4)
        self.social_attention = EnhancedAttentionLayer(embed_dim, num_heads=4)
        
        # 最终输出层
        final_dim = embed_dim * 3  # 基础嵌入 + 用户物品注意力嵌入 + 社交注意力嵌入
        self.output_layer = nn.Sequential(
            nn.Linear(final_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # 损失相关参数
        self.reg_weight = nn.Parameter(torch.tensor(0.01))
        self.social_weight = nn.Parameter(torch.tensor(0.1))
    
    def _device_sync(self, x):
        """确保特征在正确的设备上"""
        device = self.user_embedding.device
        if not hasattr(self, '_features_synced'):
            self._features_synced = True
            if hasattr(self, 'svd_user_features') and self.use_traditional_features:
                try:
                    self.svd_user_features = self.svd_user_features.to(device)
                    self.svd_item_features = self.svd_item_features.to(device)
                    self.cf_user_features = self.cf_user_features.to(device)
                    self.cf_item_features = self.cf_item_features.to(device)
                    self.stats_user_features = self.stats_user_features.to(device)
                    self.stats_item_features = self.stats_item_features.to(device)
                except Exception as e:
                    print(f"将特征移至设备时发生错误: {e}")
                    self.use_traditional_features = False
            
            if hasattr(self, 'ext_user_features') and self.use_user_features:
                self.ext_user_features = self.ext_user_features.to(device)
                
            if hasattr(self, 'ext_item_features') and self.use_item_features:
                self.ext_item_features = self.ext_item_features.to(device)
    
    def get_user_representations(self, users):
        """获取用户的完整表示"""
        self._device_sync(users)
        
        # 基础嵌入
        user_emb = self.user_embedding[users]
        reps = [user_emb]
        
        # 传统算法特征
        if hasattr(self, 'svd_user_features') and self.use_traditional_features:
            try:
                svd_emb = self.svd_processor(self.svd_user_features[users])
                cf_emb = self.cf_processor(self.cf_user_features[users])
                stats_emb = self.stats_processor(self.stats_user_features[users])
                reps.extend([svd_emb, cf_emb, stats_emb])
            except Exception as e:
                print(f"处理用户传统特征时发生错误: {e}")
        
        # 外部特征
        if self.use_user_features:
            ext_emb = self.ext_user_processor(self.ext_user_features[users])
            reps.append(ext_emb)
            
        # 拼接所有表示
        return torch.cat(reps, dim=1)
    
    def get_item_representations(self, items):
        """获取物品的完整表示"""
        self._device_sync(items)
        
        # 基础嵌入
        item_emb = self.item_embedding[items]
        reps = [item_emb]
        
        # 传统算法特征
        if hasattr(self, 'svd_item_features') and self.use_traditional_features:
            try:
                svd_emb = self.svd_processor(self.svd_item_features[items])
                cf_emb = self.cf_processor(self.cf_item_features[items])
                stats_emb = self.stats_processor(self.stats_item_features[items])
                reps.extend([svd_emb, cf_emb, stats_emb])
            except Exception as e:
                print(f"处理物品传统特征时发生错误: {e}")
        
        # 外部特征
        if self.use_item_features:
            ext_emb = self.ext_item_processor(self.ext_item_features[items])
            reps.append(ext_emb)
            
        # 拼接所有表示
        return torch.cat(reps, dim=1)
    
    def forward(self, users: torch.LongTensor, pos_items: torch.LongTensor, 
                neg_items: Optional[torch.LongTensor], ui_adj_mat: Dict[str, torch.sparse.FloatTensor], 
                social_adj_mat: Optional[torch.sparse.FloatTensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            users: 用户ID [batch_size]
            pos_items: 正样本物品ID [batch_size]
            neg_items: 负样本物品ID [batch_size, num_negs] (可选，预测时为None)
            ui_adj_mat: 用户-物品邻接矩阵
            social_adj_mat: 社交邻接矩阵 (可选)
            
        Returns:
            包含模型输出的字典
        """
        # 判断是训练模式还是预测模式
        is_prediction = neg_items is None
        
        # 1. 处理用户表示
        # 获取完整用户表示
        batch_user_rep = self.get_user_representations(users)
        all_user_rep = self.get_user_representations(torch.arange(self.num_users, device=users.device))
        
        # 通过双塔结构处理用户表示
        batch_user_emb = self.user_tower(batch_user_rep)
        all_user_emb = self.user_tower(all_user_rep)
        
        # 2. 处理物品表示
        if is_prediction:
            # 预测模式：处理所有物品
            item_idxs = torch.arange(self.num_items, device=users.device)
            all_item_rep = self.get_item_representations(item_idxs)
            all_item_emb = self.item_tower(all_item_rep)
            pos_item_emb = all_item_emb
            neg_item_emb = None
        else:
            # 训练模式：只处理正负样本物品
            pos_item_rep = self.get_item_representations(pos_items)
            pos_item_emb = self.item_tower(pos_item_rep)
            
            # 处理负样本 (可能有多个)
            if neg_items.dim() > 1:
                # 多个负样本的情况
                batch_size, num_negs = neg_items.shape
                # 将负样本展平处理
                flat_neg_items = neg_items.reshape(-1)
                
                # 获取负样本表示并处理
                flat_neg_rep = self.get_item_representations(flat_neg_items)
                flat_neg_emb = self.item_tower(flat_neg_rep)
                
                # 恢复原始形状
                neg_item_emb = flat_neg_emb.reshape(batch_size, num_negs, -1)
            else:
                # 单个负样本的情况
                neg_item_rep = self.get_item_representations(neg_items)
                neg_item_emb = self.item_tower(neg_item_rep)
        
            # 物品嵌入用于图卷积
            item_idxs = torch.arange(self.num_items, device=users.device)
            all_item_rep = self.get_item_representations(item_idxs)
            all_item_emb = self.item_tower(all_item_rep)
            
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
            scores = scores / self.temperature  # 应用温度系数
            return {
                "scores": scores
            }
        else:
            # 训练模式：计算正负样本的得分
            if neg_item_emb.dim() > 2:
                # 多个负样本情况
                # 计算正样本得分: [batch_size]
                pos_scores = torch.sum(final_user_emb.unsqueeze(1) * pos_item_emb.unsqueeze(1), dim=2).squeeze(1)
                
                # 计算负样本得分: [batch_size, num_negs]
                # 将用户嵌入扩展为 [batch_size, 1, embed_dim] 便于广播
                user_emb_expanded = final_user_emb.unsqueeze(1)
                neg_scores = torch.sum(user_emb_expanded * neg_item_emb, dim=2)
            else:
                # 单个负样本
                pos_scores = torch.sum(final_user_emb * pos_item_emb, dim=1)  # [batch_size]
                neg_scores = torch.sum(final_user_emb * neg_item_emb, dim=1)  # [batch_size]
            
            # 应用温度系数
            pos_scores = pos_scores / self.temperature
            neg_scores = neg_scores / self.temperature
            
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
        # 判断是否为预测模式
        if "scores" in batch_output:
            return {"scores": batch_output["scores"]}
            
        pos_scores = batch_output["pos_scores"]
        neg_scores = batch_output["neg_scores"]
        user_emb = batch_output["user_emb"]
        pos_item_emb = batch_output["pos_item_emb"]
        neg_item_emb = batch_output["neg_item_emb"]
        ui_emb = batch_output["ui_emb"]
        social_emb = batch_output["social_emb"]
        
        # 处理多负样本情况
        if neg_scores.dim() > 1:
            # 对每个正样本，使用所有负样本计算BPR损失
            batch_size, num_negs = neg_scores.shape
            
            # 修复维度问题 - 不使用expand_as
            # 将正样本分数重复num_negs次
            pos_scores_expanded = pos_scores.unsqueeze(1).expand(batch_size, num_negs).reshape(-1)
            neg_scores_flat = neg_scores.reshape(-1)
            
            # BPR损失
            bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores_expanded - neg_scores_flat) + 1e-10))
            
            # 对负样本嵌入求平均，用于正则化
            if neg_item_emb.dim() > 2:
                neg_item_emb_avg = torch.mean(neg_item_emb, dim=1)
            else:
                neg_item_emb_avg = neg_item_emb
        else:
            # 单负样本情况
            bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
            neg_item_emb_avg = neg_item_emb
        
        # L2正则化
        reg_loss = (torch.norm(user_emb, p=2)**2 + 
                   torch.norm(pos_item_emb, p=2)**2 + 
                   torch.norm(neg_item_emb_avg, p=2)**2)
        reg_loss = reg_loss / user_emb.shape[0]  # 归一化
        
        # 社交损失 (如果有社交嵌入)
        social_loss = torch.tensor(0.0, device=user_emb.device)
        if "social_emb" in batch_output:
            ui_diff = torch.norm(ui_emb - social_emb, p=2)**2
            social_loss = ui_diff / user_emb.shape[0]
        
        # 总损失
        reg_weight = torch.clamp(self.reg_weight, min=0.0, max=0.1)  # 限制范围
        social_weight = torch.clamp(self.social_weight, min=0.0, max=0.5)  # 限制范围
        
        total_loss = bpr_loss + reg_weight * reg_loss + social_weight * social_loss
        
        return {
            "total_loss": total_loss,
            "bpr_loss": bpr_loss,
            "reg_loss": reg_loss,
            "social_loss": social_loss,
            "reg_weight": reg_weight,
            "social_weight": social_weight
        } 