import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple, Optional

class EnhancedSVD(nn.Module):
    """SVD嵌入增强模块，为用户和物品生成高质量初始嵌入"""
    
    def __init__(self, interaction_matrix, embedding_dim):
        """
        初始化SVD嵌入
        
        参数:
            interaction_matrix: 用户-物品交互矩阵
            embedding_dim: 嵌入维度
        """
        super(EnhancedSVD, self).__init__()
        
        # 使用SVD分解交互矩阵
        from scipy.sparse.linalg import svds
        
        print(f"计算SVD分解 (dim={embedding_dim})...")
        # 确保交互矩阵是浮点型
        if interaction_matrix.dtype != np.float32 and interaction_matrix.dtype != np.float64:
            interaction_matrix = interaction_matrix.astype(np.float32)
        
        # 执行SVD
        u, s, vt = svds(interaction_matrix, k=embedding_dim)
        
        # 计算加权嵌入
        s_sqrt = np.sqrt(s)
        user_emb = u * s_sqrt[None, :]
        item_emb = vt.T * s_sqrt[None, :]
        
        # 转换为模型参数
        self.user_embedding = nn.Parameter(torch.FloatTensor(user_emb))
        self.item_embedding = nn.Parameter(torch.FloatTensor(item_emb))
        
        # 投影层 - 允许进一步调整嵌入
        self.user_projection = nn.Linear(embedding_dim, embedding_dim)
        self.item_projection = nn.Linear(embedding_dim, embedding_dim)
        
        print(f"SVD初始化完成 | 用户嵌入: {self.user_embedding.shape}, 物品嵌入: {self.item_embedding.shape}")
    
    def forward(self, user_ids=None, item_ids=None):
        """获取用户和物品的SVD嵌入"""
        if user_ids is None:
            user_emb = self.user_embedding
        else:
            user_emb = self.user_embedding[user_ids]
            
        if item_ids is None:
            item_emb = self.item_embedding
        else:
            item_emb = self.item_embedding[item_ids]
        
        # 应用投影
        user_emb = self.user_projection(user_emb)
        item_emb = self.item_projection(item_emb)
        
        return user_emb, item_emb


class FeatureProcessor(nn.Module):
    """统计特征处理模块，提取并转换用户和物品特征"""
    
    def __init__(self, feature_dim, output_dim, dropout=0.1):
        """
        初始化统计特征处理器
        
        参数:
            feature_dim: 特征维度
            output_dim: 输出维度
            dropout: Dropout率
        """
        super(FeatureProcessor, self).__init__()
        
        # 特征转换网络
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, features):
        """转换特征"""
        # 检查特征合法性
        if torch.isnan(features).any():
            features = torch.nan_to_num(features, nan=0.0)
        
        # 返回转换后的特征
        return self.feature_mlp(features)


class DualGraphConv(nn.Module):
    """双图卷积模块，同时处理用户-物品交互图和社交网络图"""
    
    def __init__(self, embed_dim, n_layers=2, dropout=0.1):
        """
        初始化双图卷积模块
        
        参数:
            embed_dim: 嵌入维度
            n_layers: 图卷积层数
            dropout: Dropout率
        """
        super(DualGraphConv, self).__init__()
        
        # 交互图卷积层
        self.ui_layers = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(n_layers)
        ])
        
        # 社交图卷积层
        self.social_layers = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(n_layers)
        ])
        
        # 层归一化
        self.layer_norms_ui = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(n_layers)
        ])
        
        self.layer_norms_social = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(n_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 层数
        self.n_layers = n_layers
    
    def forward(self, user_emb, item_emb, ui_adj_mat, social_adj_mat):
        """
        前向传播
        
        参数:
            user_emb: 用户嵌入
            item_emb: 物品嵌入
            ui_adj_mat: 用户-物品邻接矩阵
            social_adj_mat: 社交邻接矩阵
            
        返回:
            ui_embeddings: 用户-物品图中的每层用户嵌入
            social_embeddings: 社交图中的每层用户嵌入
        """
        # 用户-物品图卷积
        ui_embeddings = [user_emb]
        current_user_emb = user_emb
        current_item_emb = item_emb
        
        for i in range(self.n_layers):
            # 聚合物品信息到用户
            user_emb_agg = torch.sparse.mm(ui_adj_mat["user_item"], current_item_emb)
            user_emb_agg = self.ui_layers[i](user_emb_agg)
            user_emb_agg = F.leaky_relu(user_emb_agg)
            
            # 聚合用户信息到物品
            item_emb_agg = torch.sparse.mm(ui_adj_mat["item_user"], current_user_emb)
            item_emb_agg = self.ui_layers[i](item_emb_agg)
            item_emb_agg = F.leaky_relu(item_emb_agg)
            
            # 残差连接 + 层归一化
            current_user_emb = self.layer_norms_ui[i](current_user_emb + user_emb_agg)
            current_item_emb = self.layer_norms_ui[i](current_item_emb + item_emb_agg)
            
            # Dropout
            current_user_emb = self.dropout(current_user_emb)
            current_item_emb = self.dropout(current_item_emb)
            
            # 保存每层嵌入
            ui_embeddings.append(current_user_emb)
        
        # 社交图卷积
        social_embeddings = [user_emb]
        current_social_emb = user_emb
        
        for i in range(self.n_layers):
            # 检查维度是否匹配
            if social_adj_mat.shape[0] != current_social_emb.shape[0]:
                print(f"警告: 社交矩阵维度 {social_adj_mat.shape[0]} 与嵌入维度 {current_social_emb.shape[0]} 不匹配")
                # 创建一个适配当前批次的社交嵌入副本
                adjusted_social_emb = torch.zeros(social_adj_mat.shape[0], current_social_emb.shape[1], 
                                                 device=current_social_emb.device)
                # 如果社交矩阵比嵌入小，只使用嵌入的子集
                if social_adj_mat.shape[0] < current_social_emb.shape[0]:
                    adjusted_social_emb = current_social_emb[:social_adj_mat.shape[0]]
                else:
                    # 如果社交矩阵比嵌入大，填充现有嵌入并随机初始化剩余部分
                    adjusted_social_emb[:current_social_emb.shape[0]] = current_social_emb
                    if social_adj_mat.shape[0] > current_social_emb.shape[0]:
                        remaining = social_adj_mat.shape[0] - current_social_emb.shape[0]
                        adjusted_social_emb[current_social_emb.shape[0]:] = torch.randn(
                            remaining, current_social_emb.shape[1], device=current_social_emb.device) * 0.1
                
                current_social_emb = adjusted_social_emb
            
            # 聚合社交邻居信息
            social_emb_agg = torch.sparse.mm(social_adj_mat, current_social_emb)
            social_emb_agg = self.social_layers[i](social_emb_agg)
            social_emb_agg = F.leaky_relu(social_emb_agg)
            
            # 残差连接 + 层归一化
            current_social_emb = self.layer_norms_social[i](current_social_emb + social_emb_agg)
            
            # Dropout
            current_social_emb = self.dropout(current_social_emb)
            
            # 保存每层嵌入
            social_embeddings.append(current_social_emb)
        
        return {
            "ui_embeddings": ui_embeddings,
            "social_embeddings": social_embeddings,
            "item_embedding": current_item_emb
        }


class DualSampling(nn.Module):
    """双采样模块，用于采样社交关系和物品关系"""
    
    def __init__(self, embed_dim, temperature=0.2):
        """
        初始化双采样模块
        
        参数:
            embed_dim: 嵌入维度
            temperature: Gumbel-Softmax温度参数
        """
        super(DualSampling, self).__init__()
        
        self.temperature = temperature
        self.similarity_projection = nn.Linear(embed_dim, embed_dim)
    
    def gumbel_softmax_sampling(self, logits, tau=1.0, hard=False):
        """
        Gumbel-Softmax采样，将分类分布转换为连续近似
        
        参数:
            logits: 对数概率 [batch_size, n_class]
            tau: 温度参数
            hard: 是否生成离散的one-hot样本
        """
        # 生成Gumbel噪声
        gumbels = -torch.empty_like(logits).exponential_().log()
        
        # ~Gumbel(logits, tau)
        gumbels = (logits + gumbels) / tau 
        
        # Softmax
        y_soft = gumbels.softmax(dim=-1)
        
        if hard:
            # Straight through trick
            index = y_soft.max(dim=-1, keepdim=True)[1]
            y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
            return y_hard - y_soft.detach() + y_soft
        
        return y_soft
    
    def sample_social_relations(self, user_emb, social_adj, max_samples=10):
        """
        采样社交关系
        
        参数:
            user_emb: 用户嵌入
            social_adj: 社交邻接矩阵
            max_samples: 每个用户最大采样数
        """
        # 计算用户嵌入相似度
        user_emb_projected = self.similarity_projection(user_emb)
        sim_scores = torch.matmul(user_emb_projected, user_emb_projected.t())
        
        # 应用温度缩放
        sim_scores = sim_scores / self.temperature
        
        # 对每个用户采样
        batch_size = user_emb.size(0)
        
        # 创建掩码以排除已有的社交连接和自身
        mask = torch.eye(batch_size, device=user_emb.device)
        if social_adj is not None and social_adj._nnz() > 0:
            # 将已有社交连接添加到掩码中
            indices = social_adj._indices()
            for i in range(indices.size(1)):
                u, v = indices[0, i].item(), indices[1, i].item()
                if u < batch_size and v < batch_size:
                    mask[u, v] = 1
        
        # 将掩码应用到相似度分数
        sim_scores = sim_scores.masked_fill(mask > 0, -1e9)
        
        # Gumbel-Softmax采样
        sampled_scores = self.gumbel_softmax_sampling(sim_scores, tau=self.temperature, hard=False)
        
        # 取top-k
        _, topk_indices = torch.topk(sampled_scores, k=min(max_samples, batch_size-1), dim=1)
        
        # 构建新的采样社交关系
        new_relations = torch.zeros_like(sim_scores)
        for i in range(batch_size):
            new_relations[i, topk_indices[i]] = 1.0
        
        return new_relations
    
    def forward(self, user_emb, item_emb, social_adj=None):
        """
        前向传播，执行双采样
        
        参数:
            user_emb: 用户嵌入
            item_emb: 物品嵌入
            social_adj: 社交邻接矩阵
        """
        # 对社交关系进行采样
        sampled_social = self.sample_social_relations(user_emb, social_adj)
        
        return {
            "sampled_social_relations": sampled_social
        }


class LayerwiseAttention(nn.Module):
    """层次注意力模块，聚合不同层的嵌入"""
    
    def __init__(self, embed_dim, num_heads=1):
        """
        初始化层次注意力模块
        
        参数:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
        """
        super(LayerwiseAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        if num_heads == 1:
            # 单头注意力
            self.query = nn.Linear(embed_dim, embed_dim)
            self.key = nn.Linear(embed_dim, embed_dim)
            self.value = nn.Linear(embed_dim, embed_dim)
        else:
            # 多头注意力
            self.mha = nn.MultiheadAttention(embed_dim, num_heads)
    
    def forward(self, layer_embeddings):
        """
        前向传播
        
        参数:
            layer_embeddings: 各层嵌入列表 [n_layers+1, batch_size, embed_dim]
            
        返回:
            聚合后的嵌入 [batch_size, embed_dim]
        """
        # 检查输入的有效性
        if not layer_embeddings or len(layer_embeddings) == 0:
            raise ValueError("层嵌入列表不能为空")
        
        # 获取输入的第一个维度（用户数）
        num_users = layer_embeddings[0].shape[0]
        
        # 确保所有层的嵌入具有相同的用户数
        for i, emb in enumerate(layer_embeddings[1:], 1):
            if emb.shape[0] != num_users:
                print(f"警告: 第{i}层嵌入形状 {emb.shape} 与第0层 {layer_embeddings[0].shape} 不一致")
                # 创建新的嵌入以匹配第一层的形状
                adj_emb = torch.zeros(num_users, self.embed_dim, device=emb.device)
                # 复制有效部分
                min_size = min(num_users, emb.shape[0])
                adj_emb[:min_size] = emb[:min_size]
                # 替换原始嵌入
                layer_embeddings[i] = adj_emb
        
        # 堆叠嵌入为3D张量
        stacked_embs = torch.stack(layer_embeddings, dim=0)  # [n_layers+1, batch_size, embed_dim]
        
        if self.num_heads == 1:
            # 单头自注意力
            query = self.query(stacked_embs.mean(dim=0, keepdim=True))  # [1, batch_size, embed_dim]
            key = self.key(stacked_embs)  # [n_layers+1, batch_size, embed_dim]
            value = self.value(stacked_embs)  # [n_layers+1, batch_size, embed_dim]
            
            # 计算注意力分数
            scores = torch.matmul(query, key.transpose(-2, -1)) / (self.embed_dim ** 0.5)  # [1, batch_size, n_layers+1]
            attention_weights = F.softmax(scores, dim=-1)  # [1, batch_size, n_layers+1]
            
            # 应用注意力
            weighted_sum = torch.matmul(attention_weights, value)  # [1, batch_size, embed_dim]
            output = weighted_sum.squeeze(0)  # [batch_size, embed_dim]
        else:
            # 多头注意力
            # 调整维度顺序为[batch_size, n_layers+1, embed_dim]
            stacked_embs = stacked_embs.permute(1, 0, 2)
            batch_size, n_layers, _ = stacked_embs.shape
            
            # 创建查询 - 使用平均嵌入
            query = stacked_embs.mean(dim=1, keepdim=True)  # [batch_size, 1, embed_dim]
            
            # 应用多头注意力
            output, _ = self.mha(query, stacked_embs, stacked_embs)  # [batch_size, 1, embed_dim]
            output = output.squeeze(1)  # [batch_size, embed_dim]
        
        # 确保输出的维度正确
        if len(output.shape) != 2:
            print(f"警告: 注意力输出维度不是2: {output.shape}，尝试调整")
            if len(output.shape) == 3 and output.shape[0] == 1:
                output = output.squeeze(0)
            elif len(output.shape) == 3 and output.shape[1] == 1:
                output = output.squeeze(1)
        
        print(f"注意力输出形状: {output.shape}")
        return output


class CollaborativeFiltering(nn.Module):
    """协同过滤模块，融合嵌入相似度进行推荐"""
    
    def __init__(self, embed_dim, dropout=0.1):
        """
        初始化协同过滤模块
        
        参数:
            embed_dim: 嵌入维度
            dropout: Dropout率
        """
        super(CollaborativeFiltering, self).__init__()
        
        # 相似度投影
        self.similarity_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )
        
        # 相似度权重 - 可学习参数
        self.similarity_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, user_emb, item_emb, ui_interactions=None):
        """
        前向传播
        
        参数:
            user_emb: 用户嵌入 [batch_size, embed_dim]
            item_emb: 物品嵌入 [num_items, embed_dim] 或 [batch_size, embed_dim]
            ui_interactions: 用户-物品交互矩阵 (稀疏)
            
        返回:
            预测分数
        """
        # 投影用户嵌入
        user_emb_proj = self.similarity_projection(user_emb)
        
        # 根据物品嵌入的形状确定是计算用户-物品相似度还是用户-用户相似度
        if len(item_emb.shape) == 2 and item_emb.shape[0] != user_emb.shape[0]:
            # 用户-物品相似度
            cf_scores = torch.matmul(user_emb_proj, item_emb.t())  # [batch_size, num_items]
        else:
            # 用户-物品对的相似度
            cf_scores = torch.sum(user_emb_proj * item_emb, dim=1)  # [batch_size]
        
        return cf_scores


class SocialRecommender(nn.Module):
    """集成社交信息的推荐系统"""
    
    def __init__(self, 
                 num_users, 
                 num_items, 
                 embed_dim=64,
                 interaction_matrix=None,
                 user_features=None,
                 item_features=None,
                 n_layers=2,
                 dropout=0.1,
                 temperature=0.2):
        """
        初始化社交推荐模型
        
        参数:
            num_users: 用户数量
            num_items: 物品数量
            embed_dim: 嵌入维度
            interaction_matrix: 用户-物品交互矩阵
            user_features: 用户统计特征
            item_features: 物品统计特征
            n_layers: 图卷积层数
            dropout: Dropout率
            temperature: 温度参数
        """
        super(SocialRecommender, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        
        # 1. SVD嵌入初始化 - 如果有交互矩阵
        if interaction_matrix is not None:
            self.svd = EnhancedSVD(interaction_matrix, embed_dim)
            # 从SVD初始化用户和物品嵌入
            with torch.no_grad():
                user_emb, item_emb = self.svd.forward()
            # 创建可训练的嵌入
            self.user_embedding = nn.Parameter(user_emb.clone())
            self.item_embedding = nn.Parameter(item_emb.clone())
        else:
            # 随机初始化
            self.user_embedding = nn.Parameter(torch.randn(num_users, embed_dim) * 0.1)
            self.item_embedding = nn.Parameter(torch.randn(num_items, embed_dim) * 0.1)
            self.svd = None
        
        # 2. 统计特征处理
        self.use_user_features = user_features is not None
        self.use_item_features = item_features is not None
        
        if self.use_user_features:
            user_feature_dim = user_features.shape[1]
            self.user_feature_processor = FeatureProcessor(user_feature_dim, embed_dim, dropout)
            # 注册为缓冲区，不需要训练
            self.register_buffer('user_features', torch.FloatTensor(user_features))
        
        if self.use_item_features:
            item_feature_dim = item_features.shape[1]
            self.item_feature_processor = FeatureProcessor(item_feature_dim, embed_dim, dropout)
            # 注册为缓冲区，不需要训练
            self.register_buffer('item_features', torch.FloatTensor(item_features))
        
        # 3. 双图卷积
        self.dual_gcn = DualGraphConv(embed_dim, n_layers, dropout)
        
        # 4. 双采样
        self.dual_sampling = DualSampling(embed_dim, temperature)
        
        # 5. 层次注意力
        self.ui_attention = LayerwiseAttention(embed_dim)
        self.social_attention = LayerwiseAttention(embed_dim)
        
        # 6. 协同过滤
        self.cf = CollaborativeFiltering(embed_dim, dropout)
        
        # 7. 最终输出层 - 融合各路信息
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # 损失平衡权重
        self.reg_weight = nn.Parameter(torch.tensor(0.01))
        self.cf_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, users, pos_items, neg_items, ui_adj_mat, social_adj_mat):
        """
        前向传播
        
        参数:
            users: 用户ID
            pos_items: 正样本物品ID
            neg_items: 负样本物品ID
            ui_adj_mat: 用户-物品邻接矩阵
            social_adj_mat: 社交邻接矩阵
            
        返回:
            包含预测和中间结果的字典
        """
        # 1. 获取嵌入
        batch_user_emb = self.user_embedding[users]
        all_user_emb = self.user_embedding
        all_item_emb = self.item_embedding
        pos_item_emb = self.item_embedding[pos_items]
        neg_item_emb = self.item_embedding[neg_items]
        
        # 2. 添加统计特征（如果有）
        if self.use_user_features:
            user_feature_emb = self.user_feature_processor(self.user_features[users])
            batch_user_emb = batch_user_emb + user_feature_emb
            
            # 为所有用户计算特征嵌入
            all_user_feature_emb = self.user_feature_processor(self.user_features)
            all_user_emb = all_user_emb + all_user_feature_emb
        
        if self.use_item_features:
            pos_feature_emb = self.item_feature_processor(self.item_features[pos_items])
            pos_item_emb = pos_item_emb + pos_feature_emb
            
            neg_feature_emb = self.item_feature_processor(self.item_features[neg_items])
            neg_item_emb = neg_item_emb + neg_feature_emb
            
            # 为所有物品计算特征嵌入
            all_item_feature_emb = self.item_feature_processor(self.item_features)
            all_item_emb = all_item_emb + all_item_feature_emb
        
        # 3. 双图卷积传播
        gcn_results = self.dual_gcn(all_user_emb, all_item_emb, ui_adj_mat, social_adj_mat)
        ui_embeddings = gcn_results["ui_embeddings"]
        social_embeddings = gcn_results["social_embeddings"] if "social_embeddings" in gcn_results else None
        
        # 确保社交嵌入和用户嵌入维度一致
        if social_embeddings is not None:
            # 检查社交嵌入的维度是否与用户嵌入一致
            for i in range(len(social_embeddings)):
                if social_embeddings[i].shape[0] != self.num_users:
                    print(f"警告: 社交嵌入 {i} 维度 {social_embeddings[i].shape[0]} 与用户数 {self.num_users} 不匹配")
                    # 创建一个调整后的社交嵌入以匹配用户数
                    adjusted_emb = torch.zeros(self.num_users, self.embed_dim, device=social_embeddings[i].device)
                    # 复制有效部分
                    min_size = min(adjusted_emb.shape[0], social_embeddings[i].shape[0])
                    adjusted_emb[:min_size] = social_embeddings[i][:min_size]
                    social_embeddings[i] = adjusted_emb
        
        # 如果社交嵌入为None，使用用户-物品嵌入代替
        if social_embeddings is None:
            print("警告: 社交嵌入为None，使用用户-物品嵌入替代")
            social_embeddings = ui_embeddings
            
        final_item_emb = gcn_results["item_embedding"] if "item_embedding" in gcn_results else all_item_emb
        
        # 4. 双采样
        sampling_results = self.dual_sampling(batch_user_emb, pos_item_emb, social_adj_mat)
        sampled_social = sampling_results["sampled_social_relations"]
        
        # 5. 层次注意力聚合
        ui_emb = self.ui_attention(ui_embeddings)
        social_emb = self.social_attention(social_embeddings)
        
        # 记录注意力输出形状，以便调试
        print(f"注意力输出形状 - ui_emb: {ui_emb.shape}, social_emb: {social_emb.shape}")
        
        # 处理注意力输出 - 确保是2维张量 [users, embed_dim]
        if len(ui_emb.shape) == 3:
            print(f"发现3维ui_emb，应用均值池化: {ui_emb.shape}")
            # 如果是 [num_heads, users, embed_dim]，沿头维度取均值
            ui_emb = ui_emb.mean(dim=0)
        
        if len(social_emb.shape) == 3:
            print(f"发现3维social_emb，应用均值池化: {social_emb.shape}")
            # 如果是 [num_heads, users, embed_dim]，沿头维度取均值
            social_emb = social_emb.mean(dim=0)
        
        # 确保维度正确
        if len(ui_emb.shape) != 2 or len(social_emb.shape) != 2:
            raise ValueError(f"注意力输出维度错误 - ui_emb: {ui_emb.shape}, social_emb: {social_emb.shape}")
            
        # 提取当前批次用户的图卷积嵌入
        # 确保用户ID在有效范围内
        valid_users = torch.clamp(users, 0, ui_emb.shape[0] - 1)
        if not torch.equal(users, valid_users):
            print(f"警告: 发现无效的用户ID，已将其限制在有效范围内")
            invalid_count = (users != valid_users).sum().item()
            print(f"  - 无效ID数量: {invalid_count}")
            print(f"  - 用户嵌入形状: {ui_emb.shape}")
            print(f"  - 最大用户ID: {users.max().item()}")
            
        batch_ui_emb = ui_emb[valid_users]
        batch_social_emb = social_emb[valid_users]
        
        # 6. 协同过滤预测
        cf_pos_scores = self.cf(batch_user_emb, pos_item_emb)
        cf_neg_scores = self.cf(batch_user_emb, neg_item_emb)
        
        # 7. 最终融合
        # 融合协同过滤、用户-物品图和社交图的嵌入
        print(f"DEBUG: 连接前张量形状 - batch_user_emb: {batch_user_emb.shape}, batch_ui_emb: {batch_ui_emb.shape}, batch_social_emb: {batch_social_emb.shape}")
        
        # 确保所有张量都是2维的 [batch_size, embed_dim]
        # 如果维度不匹配，尝试处理
        try:
            if len(batch_ui_emb.shape) == 3:
                # 如果是 [batch_size, something, embed_dim]，对中间维度取均值
                print(f"batch_ui_emb 是3维的，应用均值池化: {batch_ui_emb.shape}")
                batch_ui_emb = batch_ui_emb.mean(dim=1)
            elif len(batch_ui_emb.shape) != 2:
                raise ValueError(f"batch_ui_emb 维度不正确: {batch_ui_emb.shape}")
                
            if len(batch_social_emb.shape) == 3:
                # 如果是 [batch_size, something, embed_dim]，对中间维度取均值
                print(f"batch_social_emb 是3维的，应用均值池化: {batch_social_emb.shape}")
                batch_social_emb = batch_social_emb.mean(dim=1)
            elif len(batch_social_emb.shape) != 2:
                raise ValueError(f"batch_social_emb 维度不正确: {batch_social_emb.shape}")
                
            # 检查嵌入维度是否一致
            if batch_user_emb.shape[1] != self.embed_dim or batch_ui_emb.shape[1] != self.embed_dim or batch_social_emb.shape[1] != self.embed_dim:
                raise ValueError(f"嵌入维度不一致 - user: {batch_user_emb.shape[1]}, ui: {batch_ui_emb.shape[1]}, social: {batch_social_emb.shape[1]}, expected: {self.embed_dim}")
                
            # 检查批次大小是否一致
            if batch_user_emb.shape[0] != batch_ui_emb.shape[0] or batch_user_emb.shape[0] != batch_social_emb.shape[0]:
                print(f"警告: 批次大小不一致 - user: {batch_user_emb.shape[0]}, ui: {batch_ui_emb.shape[0]}, social: {batch_social_emb.shape[0]}")
                # 调整批次大小为最小的一个
                min_batch = min(batch_user_emb.shape[0], batch_ui_emb.shape[0], batch_social_emb.shape[0])
                batch_user_emb = batch_user_emb[:min_batch]
                batch_ui_emb = batch_ui_emb[:min_batch]
                batch_social_emb = batch_social_emb[:min_batch]
                print(f"已调整批次大小为: {min_batch}")
            
            # 尝试连接张量
            final_user_emb = self.output_layer(
                torch.cat([batch_user_emb, batch_ui_emb, batch_social_emb], dim=1)
            )
        except Exception as e:
            print(f"错误: 无法连接张量: {e}")
            print(f"形状 - user: {batch_user_emb.shape}, ui: {batch_ui_emb.shape}, social: {batch_social_emb.shape}")
            # 尝试使用仅用户嵌入作为回退
            print("使用仅用户嵌入作为回退")
            # 调整维度以匹配期望输出
            final_user_emb = self.output_layer(
                torch.cat([batch_user_emb, batch_user_emb, batch_user_emb], dim=1)
            )
        
        # 计算最终预测分数
        pos_scores = torch.sum(final_user_emb * pos_item_emb, dim=1)
        neg_scores = torch.sum(final_user_emb * neg_item_emb, dim=1)
        
        # 融合协同过滤分数
        cf_weight = torch.sigmoid(self.cf_weight)
        final_pos_scores = cf_weight * cf_pos_scores + (1 - cf_weight) * pos_scores
        final_neg_scores = cf_weight * cf_neg_scores + (1 - cf_weight) * neg_scores
        
        return {
            "pos_scores": final_pos_scores,
            "neg_scores": final_neg_scores,
            "user_emb": batch_user_emb,
            "pos_item_emb": pos_item_emb,
            "neg_item_emb": neg_item_emb,
            "ui_emb": batch_ui_emb,
            "social_emb": batch_social_emb,
            "final_user_emb": final_user_emb,
            "sampled_social": sampled_social,
            "cf_pos_scores": cf_pos_scores,
            "cf_neg_scores": cf_neg_scores,
            "cf_weight": cf_weight
        }
    
    def calculate_loss(self, batch_output):
        """计算模型损失"""
        pos_scores = batch_output["pos_scores"]
        neg_scores = batch_output["neg_scores"]
        user_emb = batch_output["user_emb"]
        pos_item_emb = batch_output["pos_item_emb"]
        neg_item_emb = batch_output["neg_item_emb"]
        
        # BPR损失 - 主要推荐损失
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
        
        # 正则化损失 - 防止过拟合
        reg_loss = torch.norm(user_emb)**2 + torch.norm(pos_item_emb)**2 + torch.norm(neg_item_emb)**2
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
    
    def predict(self, users, ui_adj_mat, social_adj_mat):
        """为用户预测所有物品的得分"""
        # 获取用户嵌入
        batch_user_emb = self.user_embedding[users]
        all_user_emb = self.user_embedding
        all_item_emb = self.item_embedding
        
        # 添加统计特征（如果有）
        if self.use_user_features:
            user_feature_emb = self.user_feature_processor(self.user_features[users])
            batch_user_emb = batch_user_emb + user_feature_emb
            
            all_user_feature_emb = self.user_feature_processor(self.user_features)
            all_user_emb = all_user_emb + all_user_feature_emb
        
        if self.use_item_features:
            all_item_feature_emb = self.item_feature_processor(self.item_features)
            all_item_emb = all_item_emb + all_item_feature_emb
        
        # 图卷积传播
        gcn_results = self.dual_gcn(all_user_emb, all_item_emb, ui_adj_mat, social_adj_mat)
        ui_embeddings = gcn_results["ui_embeddings"]
        social_embeddings = gcn_results["social_embeddings"] if "social_embeddings" in gcn_results else None
        
        # 确保社交嵌入和用户嵌入维度一致
        if social_embeddings is not None:
            # 检查社交嵌入的维度是否与用户嵌入一致
            for i in range(len(social_embeddings)):
                if social_embeddings[i].shape[0] != self.num_users:
                    print(f"警告: 社交嵌入 {i} 维度 {social_embeddings[i].shape[0]} 与用户数 {self.num_users} 不匹配")
                    # 创建一个调整后的社交嵌入以匹配用户数
                    adjusted_emb = torch.zeros(self.num_users, self.embed_dim, device=social_embeddings[i].device)
                    # 复制有效部分
                    min_size = min(adjusted_emb.shape[0], social_embeddings[i].shape[0])
                    adjusted_emb[:min_size] = social_embeddings[i][:min_size]
                    social_embeddings[i] = adjusted_emb
        
        # 如果社交嵌入为None，使用用户-物品嵌入代替
        if social_embeddings is None:
            print("警告: 社交嵌入为None，使用用户-物品嵌入替代")
            social_embeddings = ui_embeddings
            
        final_item_emb = gcn_results["item_embedding"] if "item_embedding" in gcn_results else all_item_emb
        
        # 注意力聚合
        ui_emb = self.ui_attention(ui_embeddings)
        social_emb = self.social_attention(social_embeddings)
        
        # 记录注意力输出形状，以便调试
        print(f"注意力输出形状 - ui_emb: {ui_emb.shape}, social_emb: {social_emb.shape}")
        
        # 处理注意力输出 - 确保是2维张量 [users, embed_dim]
        if len(ui_emb.shape) == 3:
            print(f"发现3维ui_emb，应用均值池化: {ui_emb.shape}")
            # 如果是 [num_heads, users, embed_dim]，沿头维度取均值
            ui_emb = ui_emb.mean(dim=0)
        
        if len(social_emb.shape) == 3:
            print(f"发现3维social_emb，应用均值池化: {social_emb.shape}")
            # 如果是 [num_heads, users, embed_dim]，沿头维度取均值
            social_emb = social_emb.mean(dim=0)
        
        # 确保维度正确
        if len(ui_emb.shape) != 2 or len(social_emb.shape) != 2:
            raise ValueError(f"注意力输出维度错误 - ui_emb: {ui_emb.shape}, social_emb: {social_emb.shape}")
            
        # 提取当前批次用户的图卷积嵌入
        # 确保用户ID在有效范围内
        valid_users = torch.clamp(users, 0, ui_emb.shape[0] - 1)
        if not torch.equal(users, valid_users):
            print(f"警告: 发现无效的用户ID，已将其限制在有效范围内")
            invalid_count = (users != valid_users).sum().item()
            print(f"  - 无效ID数量: {invalid_count}")
            print(f"  - 用户嵌入形状: {ui_emb.shape}")
            print(f"  - 最大用户ID: {users.max().item()}")
            
        batch_ui_emb = ui_emb[valid_users]
        batch_social_emb = social_emb[valid_users]
        
        # 协同过滤预测
        cf_scores = self.cf(batch_user_emb, final_item_emb)
        
        # 最终融合
        print(f"DEBUG (predict): 连接前张量形状 - batch_user_emb: {batch_user_emb.shape}, batch_ui_emb: {batch_ui_emb.shape}, batch_social_emb: {batch_social_emb.shape}")
        
        # 确保所有张量都是2维的 [batch_size, embed_dim]
        # 如果维度不匹配，尝试处理
        try:
            if len(batch_ui_emb.shape) == 3:
                # 如果是 [batch_size, something, embed_dim]，对中间维度取均值
                print(f"batch_ui_emb 是3维的，应用均值池化: {batch_ui_emb.shape}")
                batch_ui_emb = batch_ui_emb.mean(dim=1)
            elif len(batch_ui_emb.shape) != 2:
                raise ValueError(f"batch_ui_emb 维度不正确: {batch_ui_emb.shape}")
                
            if len(batch_social_emb.shape) == 3:
                # 如果是 [batch_size, something, embed_dim]，对中间维度取均值
                print(f"batch_social_emb 是3维的，应用均值池化: {batch_social_emb.shape}")
                batch_social_emb = batch_social_emb.mean(dim=1)
            elif len(batch_social_emb.shape) != 2:
                raise ValueError(f"batch_social_emb 维度不正确: {batch_social_emb.shape}")
                
            # 检查嵌入维度是否一致
            if batch_user_emb.shape[1] != self.embed_dim or batch_ui_emb.shape[1] != self.embed_dim or batch_social_emb.shape[1] != self.embed_dim:
                raise ValueError(f"嵌入维度不一致 - user: {batch_user_emb.shape[1]}, ui: {batch_ui_emb.shape[1]}, social: {batch_social_emb.shape[1]}, expected: {self.embed_dim}")
                
            # 检查批次大小是否一致
            if batch_user_emb.shape[0] != batch_ui_emb.shape[0] or batch_user_emb.shape[0] != batch_social_emb.shape[0]:
                print(f"警告: 批次大小不一致 - user: {batch_user_emb.shape[0]}, ui: {batch_ui_emb.shape[0]}, social: {batch_social_emb.shape[0]}")
                # 调整批次大小为最小的一个
                min_batch = min(batch_user_emb.shape[0], batch_ui_emb.shape[0], batch_social_emb.shape[0])
                batch_user_emb = batch_user_emb[:min_batch]
                batch_ui_emb = batch_ui_emb[:min_batch]
                batch_social_emb = batch_social_emb[:min_batch]
                print(f"已调整批次大小为: {min_batch}")
            
            # 尝试连接张量
            final_user_emb = self.output_layer(
                torch.cat([batch_user_emb, batch_ui_emb, batch_social_emb], dim=1)
            )
        except Exception as e:
            print(f"错误: 无法连接张量: {e}")
            print(f"形状 - user: {batch_user_emb.shape}, ui: {batch_ui_emb.shape}, social: {batch_social_emb.shape}")
            # 尝试使用仅用户嵌入作为回退
            print("使用仅用户嵌入作为回退")
            # 调整维度以匹配期望输出
            final_user_emb = self.output_layer(
                torch.cat([batch_user_emb, batch_user_emb, batch_user_emb], dim=1)
            )
        
        # 计算最终得分
        scores = torch.matmul(final_user_emb, final_item_emb.t())
        
        # 融合协同过滤分数
        cf_weight = torch.sigmoid(self.cf_weight)
        final_scores = cf_weight * cf_scores + (1 - cf_weight) * scores
        
        return final_scores 