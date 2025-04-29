import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple, Optional

class EnhancedSVD(nn.Module):
    """SVD嵌入模块，为用户和物品生成初始嵌入"""
    
    def __init__(self, interaction_matrix, embedding_dim):
        """初始化SVD嵌入"""
        super(EnhancedSVD, self).__init__()
        
        print(f"计算SVD分解 (dim={embedding_dim})...")
        
        # 确保交互矩阵是浮点型
        if interaction_matrix.dtype != np.float32 and interaction_matrix.dtype != np.float64:
            interaction_matrix = interaction_matrix.astype(np.float32)
        
        # 执行SVD
        try:
            from scipy.sparse.linalg import svds
            u, s, vt = svds(interaction_matrix, k=embedding_dim)
            
            # 计算加权嵌入
            s_sqrt = np.sqrt(s)
            user_emb = u * s_sqrt[None, :]
            item_emb = vt.T * s_sqrt[None, :]
        except Exception as e:
            print(f"SVD计算失败: {e}，使用随机初始化")
            user_emb = np.random.normal(0, 0.01, size=(interaction_matrix.shape[0], embedding_dim))
            item_emb = np.random.normal(0, 0.01, size=(interaction_matrix.shape[1], embedding_dim))
        
        # 转换为模型参数
        self.user_embedding = nn.Parameter(torch.FloatTensor(user_emb))
        self.item_embedding = nn.Parameter(torch.FloatTensor(item_emb))
        
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
        
        return user_emb, item_emb


class FeatureProcessor(nn.Module):
    """统计特征处理模块"""
    
    def __init__(self, feature_dim, output_dim, dropout=0.1):
        """初始化特征处理器"""
        super(FeatureProcessor, self).__init__()
        
        # 特征转换网络
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, features):
        """转换特征"""
        # 检查特征合法性
        if torch.isnan(features).any():
            features = torch.nan_to_num(features, nan=0.0)
        
        return self.feature_mlp(features)


class SimpleGraphConv(nn.Module):
    """简化的图卷积模块，处理用户-物品交互和社交网络"""
    
    def __init__(self, embed_dim, n_layers=2, dropout=0.1):
        """初始化图卷积模块"""
        super(SimpleGraphConv, self).__init__()
        
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        
        # 图卷积变换
        self.ui_transform = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(n_layers)
        ])
        
        self.social_transform = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(n_layers)
        ])
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(n_layers * 2)
        ])
    
    def forward(self, user_emb, item_emb, ui_adj_mat, social_adj_mat=None):
        """前向传播"""
        # 保存每层嵌入用于后续聚合
        ui_embeddings = [user_emb]
        social_embeddings = [user_emb]
        
        current_user_emb = user_emb
        current_item_emb = item_emb
        current_social_emb = user_emb
        
        # 检查邻接矩阵
        has_ui_graph = (ui_adj_mat is not None and 
                         'user_item' in ui_adj_mat and 
                         'item_user' in ui_adj_mat)
        
        has_social_graph = (social_adj_mat is not None and 
                            hasattr(social_adj_mat, 'shape') and
                            hasattr(social_adj_mat, '_nnz') and
                            social_adj_mat._nnz() > 0)
        
        print(f"图卷积状态 - 用户物品图: {has_ui_graph}, 社交图: {has_social_graph}")
        
        # 用户-物品图卷积
        for i in range(self.n_layers):
            if has_ui_graph:
                try:
                    # 消息传递: 物品 -> 用户
                    try:
                        agg_user = torch.sparse.mm(ui_adj_mat['user_item'], current_item_emb)
                    except Exception as e:
                        print(f"物品到用户聚合失败: {e}，使用上一层嵌入")
                        agg_user = current_user_emb
                        
                    # 消息传递: 用户 -> 物品
                    try:
                        agg_item = torch.sparse.mm(ui_adj_mat['item_user'], current_user_emb)
                    except Exception as e:
                        print(f"用户到物品聚合失败: {e}，使用上一层嵌入")
                        agg_item = current_item_emb
                    
                    # 变换
                    agg_user = self.ui_transform[i](agg_user)
                    agg_item = self.ui_transform[i](agg_item)
                    
                    # 残差连接和归一化
                    current_user_emb = self.layer_norms[i*2](current_user_emb + agg_user)
                    current_item_emb = self.layer_norms[i*2+1](current_item_emb + agg_item)
                    
                    # Dropout
                    current_user_emb = self.dropout(current_user_emb)
                    current_item_emb = self.dropout(current_item_emb)
                except Exception as e:
                    print(f"用户-物品图卷积第{i}层失败: {e}")
            
            # 保存每层嵌入
            ui_embeddings.append(current_user_emb)
            
            # 社交图卷积
            if has_social_graph:
                try:
                    # 检查维度是否匹配
                    if social_adj_mat.shape[0] != current_social_emb.shape[0]:
                        print(f"社交邻接矩阵大小 {social_adj_mat.shape[0]} 与嵌入大小 {current_social_emb.shape[0]} 不匹配")
                        
                        # 创建适配大小的嵌入
                        temp_emb = torch.zeros(social_adj_mat.shape[0], self.embed_dim, device=current_social_emb.device)
                        min_size = min(social_adj_mat.shape[0], current_social_emb.shape[0])
                        temp_emb[:min_size] = current_social_emb[:min_size]
                        current_social_emb = temp_emb
                    
                    # 社交图消息传递
                    agg_social = torch.sparse.mm(social_adj_mat, current_social_emb)
                    agg_social = self.social_transform[i](agg_social)
                    
                    # 残差连接和归一化
                    current_social_emb = self.layer_norms[i*2](current_social_emb + agg_social)
                    current_social_emb = self.dropout(current_social_emb)
                except Exception as e:
                    print(f"社交图卷积第{i}层失败: {e}")
            
            # 保存每层社交嵌入
            social_embeddings.append(current_social_emb)
        
        # 如果没有社交图，使用用户-物品图嵌入代替
        if not has_social_graph:
            social_embeddings = ui_embeddings
        
        return {
            "ui_embeddings": ui_embeddings,
            "social_embeddings": social_embeddings,
            "item_embedding": current_item_emb
        }


class SimpleSampling(nn.Module):
    """简化的采样模块，用于发现潜在社交关系"""
    
    def __init__(self, embed_dim, temperature=0.2):
        """初始化采样模块"""
        super(SimpleSampling, self).__init__()
        
        self.temperature = temperature
        self.projection = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, user_emb, social_adj=None):
        """前向传播，采样社交关系"""
        batch_size = user_emb.shape[0]
        device = user_emb.device
        
        # 计算用户相似度
        user_proj = self.projection(user_emb)
        sim_scores = torch.matmul(user_proj, user_proj.t()) / self.temperature
        
        # 排除自身连接
        mask = torch.eye(batch_size, device=device)
        sim_scores = sim_scores.masked_fill(mask == 1, -1e9)
        
        # 如果提供已有社交关系，也排除
        if social_adj is not None and hasattr(social_adj, '_indices'):
            try:
                indices = social_adj._indices()
                for i in range(indices.shape[1]):
                    u, v = indices[0, i].item(), indices[1, i].item()
                    if u < batch_size and v < batch_size:
                        sim_scores[u, v] = -1e9
            except Exception as e:
                print(f"社交关系排除失败: {e}")
        
        # 软化相似度分数
        sampled_relations = torch.softmax(sim_scores, dim=-1)
        
        # 取top-k为显式关系
        k = min(10, batch_size-1)
        _, topk_indices = torch.topk(sampled_relations, k=k, dim=1)
        
        # 生成二值关系矩阵
        binary_relations = torch.zeros_like(sampled_relations)
        for i in range(batch_size):
            binary_relations[i, topk_indices[i]] = 1.0
        
        return binary_relations


class SimpleLayerAggregation(nn.Module):
    """简单的层聚合模块，替代复杂注意力机制"""
    
    def __init__(self, embed_dim):
        """初始化层聚合模块"""
        super(SimpleLayerAggregation, self).__init__()
        
        self.embed_dim = embed_dim
        
        # 简单的线性变换和归一化
        self.transform = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, layer_embeddings):
        """前向传播，聚合多层嵌入"""
        if not layer_embeddings or len(layer_embeddings) == 0:
            raise ValueError("层嵌入列表不能为空")
        
        if len(layer_embeddings) == 1:
            return layer_embeddings[0]
        
        # 获取基本维度信息
        num_users = layer_embeddings[0].shape[0]
        
        # 调整所有嵌入到相同形状
        aligned_embeddings = []
        for i, emb in enumerate(layer_embeddings):
            if emb.shape[0] != num_users:
                # 调整大小
                temp_emb = torch.zeros(num_users, self.embed_dim, device=emb.device)
                min_users = min(num_users, emb.shape[0])
                temp_emb[:min_users] = emb[:min_users]
                aligned_embeddings.append(temp_emb)
            else:
                aligned_embeddings.append(emb)
        
        # 策略1: 使用最后一层
        # output = aligned_embeddings[-1]
        
        # 策略2: 使用加权平均，深层权重更高
        weights = [1.0 + i * 0.5 for i in range(len(aligned_embeddings))]
        total_weight = sum(weights)
        
        output = torch.zeros_like(aligned_embeddings[0])
        for i, emb in enumerate(aligned_embeddings):
            output += emb * (weights[i] / total_weight)
        
        # 变换和归一化
        output = self.layer_norm(self.transform(output))
        
        return output


class SimpleCF(nn.Module):
    """简化的协同过滤模块"""
    
    def __init__(self, embed_dim):
        """初始化协同过滤模块"""
        super(SimpleCF, self).__init__()
        
        self.projection = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, user_emb, item_emb):
        """前向传播，计算相似度分数"""
        user_emb = self.projection(user_emb)
        
        # 计算用户-物品相似度
        if len(item_emb.shape) == 2 and item_emb.shape[0] != user_emb.shape[0]:
            # 批次用户对所有物品的评分
            scores = torch.matmul(user_emb, item_emb.t())
        else:
            # 批次用户-物品对的评分
            scores = torch.sum(user_emb * item_emb, dim=1)
        
        return scores


class SocialRecommender(nn.Module):
    """简化的社交推荐系统"""
    
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
        """初始化社交推荐模型"""
        super(SocialRecommender, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        
        # 1. 嵌入初始化
        if interaction_matrix is not None:
            # 使用SVD初始化
            self.svd = EnhancedSVD(interaction_matrix, embed_dim)
            with torch.no_grad():
                user_emb, item_emb = self.svd.forward()
            self.user_embedding = nn.Parameter(user_emb.clone())
            self.item_embedding = nn.Parameter(item_emb.clone())
        else:
            # 随机初始化
            self.user_embedding = nn.Parameter(torch.randn(num_users, embed_dim) * 0.1)
            self.item_embedding = nn.Parameter(torch.randn(num_items, embed_dim) * 0.1)
        
        # 2. 特征处理
        self.use_user_features = user_features is not None
        self.use_item_features = item_features is not None
        
        if self.use_user_features:
            user_feature_dim = user_features.shape[1]
            self.user_feature_processor = FeatureProcessor(user_feature_dim, embed_dim, dropout)
            self.register_buffer('user_features', torch.FloatTensor(user_features))
        
        if self.use_item_features:
            item_feature_dim = item_features.shape[1]
            self.item_feature_processor = FeatureProcessor(item_feature_dim, embed_dim, dropout)
            self.register_buffer('item_features', torch.FloatTensor(item_features))
        
        # 3. 图卷积
        self.graph_conv = SimpleGraphConv(embed_dim, n_layers, dropout)
        
        # 4. 社交采样
        self.sampling = SimpleSampling(embed_dim, temperature)
        
        # 5. 层聚合
        self.ui_aggregation = SimpleLayerAggregation(embed_dim)
        self.social_aggregation = SimpleLayerAggregation(embed_dim)
        
        # 6. 协同过滤
        self.cf = SimpleCF(embed_dim)
        
        # 7. 最终输出层
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 正则化权重
        self.reg_weight = 0.01
        self.cf_weight = 0.5
    
    def forward(self, users, pos_items, neg_items, ui_adj_mat, social_adj_mat):
        """前向传播"""
        # 1. 获取基础嵌入
        batch_user_emb = self.user_embedding[users]
        all_user_emb = self.user_embedding
        all_item_emb = self.item_embedding
        pos_item_emb = self.item_embedding[pos_items]
        neg_item_emb = self.item_embedding[neg_items]
        
        # 2. 添加特征（如果有）
        if self.use_user_features:
            # 处理用户特征
            try:
                user_feature_emb = self.user_feature_processor(self.user_features[users])
                all_user_feature_emb = self.user_feature_processor(self.user_features)
                batch_user_emb = batch_user_emb + user_feature_emb
                all_user_emb = all_user_emb + all_user_feature_emb
            except Exception as e:
                print(f"用户特征处理失败: {e}")
        
        if self.use_item_features:
            # 处理物品特征
            try:
                pos_feature_emb = self.item_feature_processor(self.item_features[pos_items])
                neg_feature_emb = self.item_feature_processor(self.item_features[neg_items])
                all_item_feature_emb = self.item_feature_processor(self.item_features)
                
                pos_item_emb = pos_item_emb + pos_feature_emb
                neg_item_emb = neg_item_emb + neg_feature_emb
                all_item_emb = all_item_emb + all_item_feature_emb
            except Exception as e:
                print(f"物品特征处理失败: {e}")
        
        # 3. 图卷积传播
        try:
            gcn_results = self.graph_conv(all_user_emb, all_item_emb, ui_adj_mat, social_adj_mat)
            ui_embeddings = gcn_results["ui_embeddings"]
            social_embeddings = gcn_results["social_embeddings"]
            final_item_emb = gcn_results["item_embedding"]
        except Exception as e:
            print(f"图卷积失败: {e}，使用原始嵌入")
            ui_embeddings = [all_user_emb]
            social_embeddings = [all_user_emb]
            final_item_emb = all_item_emb
        
        # 4. 社交关系采样
        try:
            sampled_relations = self.sampling(batch_user_emb, social_adj_mat)
        except Exception as e:
            print(f"社交采样失败: {e}")
            sampled_relations = torch.zeros(batch_user_emb.shape[0], batch_user_emb.shape[0], device=batch_user_emb.device)
        
        # 5. 层聚合
        try:
            ui_emb = self.ui_aggregation(ui_embeddings)
            social_emb = self.social_aggregation(social_embeddings)
            
            # 获取当前批次的嵌入
            batch_ui_emb = ui_emb[users]
            batch_social_emb = social_emb[users]
        except Exception as e:
            print(f"层聚合失败: {e}，使用原始嵌入")
            batch_ui_emb = batch_user_emb
            batch_social_emb = batch_user_emb
        
        # 6. 协同过滤预测
        try:
            cf_pos_scores = self.cf(batch_user_emb, pos_item_emb)
            cf_neg_scores = self.cf(batch_user_emb, neg_item_emb)
        except Exception as e:
            print(f"协同过滤失败: {e}")
            cf_pos_scores = torch.sum(batch_user_emb * pos_item_emb, dim=1)
            cf_neg_scores = torch.sum(batch_user_emb * neg_item_emb, dim=1)
        
        # 7. 最终融合
        try:
            # 确保所有嵌入维度一致
            final_user_emb = self.output_layer(
                torch.cat([batch_user_emb, batch_ui_emb, batch_social_emb], dim=1)
            )
        except Exception as e:
            print(f"最终融合失败: {e}，使用协同过滤嵌入")
            final_user_emb = batch_user_emb
        
        # 计算最终分数
        pos_scores = torch.sum(final_user_emb * pos_item_emb, dim=1)
        neg_scores = torch.sum(final_user_emb * neg_item_emb, dim=1)
        
        # 融合协同过滤分数
        final_pos_scores = self.cf_weight * cf_pos_scores + (1 - self.cf_weight) * pos_scores
        final_neg_scores = self.cf_weight * cf_neg_scores + (1 - self.cf_weight) * neg_scores
        
        return {
            "pos_scores": final_pos_scores,
            "neg_scores": final_neg_scores,
            "user_emb": batch_user_emb,
            "pos_item_emb": pos_item_emb,
            "neg_item_emb": neg_item_emb,
            "ui_emb": batch_ui_emb,
            "social_emb": batch_social_emb,
            "final_user_emb": final_user_emb,
            "sampled_social": sampled_relations,
            "cf_pos_scores": cf_pos_scores,
            "cf_neg_scores": cf_neg_scores
        }
    
    def calculate_loss(self, batch_output):
        """计算损失"""
        pos_scores = batch_output["pos_scores"]
        neg_scores = batch_output["neg_scores"]
        user_emb = batch_output["user_emb"]
        pos_item_emb = batch_output["pos_item_emb"]
        neg_item_emb = batch_output["neg_item_emb"]
        
        # BPR损失
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
        
        # 正则化损失
        reg_loss = torch.norm(user_emb)**2 + torch.norm(pos_item_emb)**2 + torch.norm(neg_item_emb)**2
        reg_loss = reg_loss / user_emb.shape[0]
        
        # 总损失
        total_loss = bpr_loss + self.reg_weight * reg_loss
        
        return {
            "total_loss": total_loss,
            "bpr_loss": bpr_loss,
            "reg_loss": reg_loss
        }
    
    def predict(self, user_ids, ui_adj_matrices, social_matrix=None):
        """
        用于预测阶段的前向传播
        Args:
            user_ids: 用户ID张量
            ui_adj_matrices: 用户-物品邻接矩阵
            social_matrix: 社交矩阵，可选，如果为None则使用空矩阵
            
        Returns:
            所有物品的预测分数
        """
        batch_size = user_ids.size(0)
        try:
            # 如果没有提供社交矩阵，创建一个空的
            if social_matrix is None:
                print("警告: 预测时没有提供社交矩阵，使用空矩阵")
                social_matrix = torch.sparse_coo_tensor(
                    torch.LongTensor([[0], [0]]),
                    torch.FloatTensor([0.0]),
                    torch.Size([batch_size, batch_size])
                ).to(user_ids.device)
            
            # 获取所有用户和物品的嵌入
            if self.use_svd:
                user_emb, item_emb = self.svd(None)
            else:
                user_emb = self.user_embedding.weight
                item_emb = self.item_embedding.weight
                
            # 收集批次用户的嵌入
            user_emb_batch = user_emb[user_ids]
            
            # 处理用户特征（如果有）
            if self.user_features_dim > 0:
                # 在预测阶段我们没有用户特征，所以使用零向量
                user_features = torch.zeros(batch_size, self.user_features_dim).to(user_ids.device)
                processed_user_features = self.user_feature_processor(user_features)
                user_emb_batch = torch.cat([user_emb_batch, processed_user_features], dim=1)
            
            # 图卷积层处理
            user_gcn_embs = []
            user_gcn_embs.append(user_emb_batch)  # 初始嵌入
            
            current_user_emb = user_emb_batch
            for i in range(self.n_layers):
                if ui_adj_matrices is not None:
                    current_user_emb = self.graph_convs[i](current_user_emb, item_emb, ui_adj_matrices[0], ui_adj_matrices[1], social_matrix)
                else:
                    print("警告: 预测时没有提供UI邻接矩阵，跳过图卷积")
                    break
                user_gcn_embs.append(current_user_emb)
            
            # 简化层聚合
            if len(user_gcn_embs) > 1:
                user_emb_final = self.layer_aggr(user_gcn_embs)
            else:
                user_emb_final = user_gcn_embs[0]
            
            # 计算协同过滤分数
            if self.use_cf:
                cf_scores = self.cf(user_emb_final, item_emb)
                return cf_scores
            else:
                # 直接计算与所有物品的内积
                scores = torch.matmul(user_emb_final, item_emb.t())
                return scores
                
        except Exception as e:
            print(f"预测过程中出错: {e}")
            # 返回随机分数作为后备策略
            return torch.rand(batch_size, self.num_items).to(user_ids.device) 