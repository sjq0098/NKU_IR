import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp


class MFInitializer(nn.Module):
    """矩阵分解（MF）初始化模块"""
    def __init__(self, n_users, n_items, latent_dim, lambda_reg=0.01):
        super(MFInitializer, self).__init__()
        self.user_embeddings = nn.Embedding(n_users, latent_dim)
        self.item_embeddings = nn.Embedding(n_items, latent_dim)
        self.lambda_reg = lambda_reg
        
        # 初始化嵌入
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
    
    def forward(self, user_ids, item_ids=None):
        user_embeds = self.user_embeddings(user_ids)
        
        if item_ids is not None:
            item_embeds = self.item_embeddings(item_ids)
            return user_embeds, item_embeds
        
        return user_embeds
    
    def reg_loss(self):
        """正则化损失"""
        user_reg = torch.norm(self.user_embeddings.weight, p=2) ** 2
        item_reg = torch.norm(self.item_embeddings.weight, p=2) ** 2
        return self.lambda_reg * (user_reg + item_reg) / 2


class SVDInitializer(nn.Module):
    """截断SVD初始化模块"""
    def __init__(self, interaction_matrix, latent_dim):
        super(SVDInitializer, self).__init__()
        
        # 转换为scipy.sparse矩阵
        if not sp.issparse(interaction_matrix):
            interaction_matrix = sp.csr_matrix(interaction_matrix)
        
        # 确保矩阵是浮点型
        if interaction_matrix.dtype not in [np.float32, np.float64, np.complex64, np.complex128]:
            interaction_matrix = interaction_matrix.astype(np.float32)
        
        try:
            # 进行截断SVD分解
            u, s, vt = sp.linalg.svds(interaction_matrix, k=min(latent_dim, min(interaction_matrix.shape)-1))
            
            # 构造嵌入向量 - 修复广播错误
            s_sqrt = np.sqrt(s)
            
            # 正确的矩阵乘法，确保维度匹配
            user_matrix = u.copy()
            item_matrix = vt.T.copy()
            
            # 对每个用户/物品的潜在因子乘以对应的奇异值平方根
            for i in range(len(s)):
                user_matrix[:, i] *= s_sqrt[i]
                item_matrix[:, i] *= s_sqrt[i]
            
            self.user_embeddings = nn.Parameter(torch.FloatTensor(user_matrix))
            self.item_embeddings = nn.Parameter(torch.FloatTensor(item_matrix))
        except Exception as e:
            print(f"SVD分解失败: {e}")
            print("使用随机初始化替代")
            self.user_embeddings = nn.Parameter(torch.FloatTensor(interaction_matrix.shape[0], latent_dim).normal_(0, 0.01))
            self.item_embeddings = nn.Parameter(torch.FloatTensor(interaction_matrix.shape[1], latent_dim).normal_(0, 0.01))
    
    def forward(self, user_indices, item_indices=None):
        user_embeds = self.user_embeddings[user_indices]
        
        if item_indices is not None:
            item_embeds = self.item_embeddings[item_indices]
            return user_embeds, item_embeds
        
        return user_embeds


class StatFeatureProcessor(nn.Module):
    """统计特征处理模块"""
    def __init__(self, feature_dim, embed_dim):
        super(StatFeatureProcessor, self).__init__()
        
        # 记录维度
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.hidden_dim = embed_dim * 2
        
        # 修改处理流程，先使用线性层然后应用LayerNorm
        self.fc1 = nn.Linear(feature_dim, self.hidden_dim)
        # LayerNorm应该应用于特征维度，而不是batch维度
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, embed_dim)
        
        print(f"StatFeatureProcessor初始化: 输入维度={feature_dim}, 隐藏维度={self.hidden_dim}, 输出维度={embed_dim}")
        
    def forward(self, x):
        # 确保输入特征是float32类型
        if x.dtype != torch.float32:
            x = x.float()
        
        # 先应用第一个线性层
        h1 = self.fc1(x)
        
        # 然后应用层归一化
        h1 = self.layer_norm(h1)
        
        # 应用激活函数
        h1 = F.leaky_relu(h1)
        
        # 最后应用第二个线性层
        out = self.fc2(h1)
        
        return out


class LightGCN(nn.Module):
    """LightGCN消息传递模块"""
    def __init__(self, n_users, n_items, interaction_matrix, embed_dim, n_layers=2, alpha=0.1):
        super(LightGCN, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.alpha = alpha
        
        # 归一化用户-物品邻接矩阵
        if not sp.issparse(interaction_matrix):
            interaction_matrix = sp.csr_matrix(interaction_matrix)
        
        # 构造用户-物品双向邻接矩阵
        user_degrees = np.array(interaction_matrix.sum(axis=1)).flatten()
        item_degrees = np.array(interaction_matrix.sum(axis=0)).flatten()
        
        # 处理度为0的情况
        user_degrees[user_degrees == 0] = 1
        item_degrees[item_degrees == 0] = 1
        
        # 计算归一化邻接矩阵
        user_diag_inv = sp.diags(1 / user_degrees)
        item_diag_inv = sp.diags(1 / item_degrees)
        
        self.A_ui = user_diag_inv @ interaction_matrix
        self.A_iu = item_diag_inv @ interaction_matrix.T
        
        # 将稀疏矩阵转为PyTorch的稀疏格式
        self.A_ui_indices = torch.LongTensor(np.vstack(self.A_ui.nonzero()).astype(np.int64))
        self.A_ui_values = torch.FloatTensor(self.A_ui.data.astype(np.float32))
        self.A_iu_indices = torch.LongTensor(np.vstack(self.A_iu.nonzero()).astype(np.int64))
        self.A_iu_values = torch.FloatTensor(self.A_iu.data.astype(np.float32))
        
        # 嵌入初始化
        self.user_embeddings = nn.Embedding(n_users, embed_dim)
        self.item_embeddings = nn.Embedding(n_items, embed_dim)
        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.item_embeddings.weight)
    
    def to(self, device):
        super(LightGCN, self).to(device)
        self.A_ui_indices = self.A_ui_indices.to(device)
        self.A_ui_values = self.A_ui_values.to(device)
        self.A_iu_indices = self.A_iu_indices.to(device)
        self.A_iu_values = self.A_iu_values.to(device)
        return self
    
    def forward(self, init_user_embeds=None, init_item_embeds=None):
        # 如果没有提供初始嵌入，则使用随机初始化的嵌入
        if init_user_embeds is None:
            user_embeds_0 = self.user_embeddings.weight
        else:
            user_embeds_0 = init_user_embeds
            
        if init_item_embeds is None:
            item_embeds_0 = self.item_embeddings.weight
        else:
            item_embeds_0 = init_item_embeds
        
        # 初始层嵌入
        user_embeds = [user_embeds_0]
        item_embeds = [item_embeds_0]
        
        # 获取当前设备
        device = user_embeds_0.device
        
        # 消息传递层
        for l in range(self.n_layers):
            # 用户-物品传播 - 确保在正确的设备上创建稀疏张量
            A_ui_sparse = torch.sparse_coo_tensor(
                self.A_ui_indices, 
                self.A_ui_values,
                torch.Size([self.n_users, self.n_items]),
                device=device  # 明确指定设备
            )
            
            # 物品-用户传播 - 确保在正确的设备上创建稀疏张量
            A_iu_sparse = torch.sparse_coo_tensor(
                self.A_iu_indices, 
                self.A_iu_values,
                torch.Size([self.n_items, self.n_users]),
                device=device  # 明确指定设备
            )
            
            # 计算新的嵌入
            user_embeds_new = torch.sparse.mm(A_ui_sparse, item_embeds[-1]) + self.alpha * user_embeds_0
            item_embeds_new = torch.sparse.mm(A_iu_sparse, user_embeds[-1]) + self.alpha * item_embeds_0
            
            # 添加到嵌入列表
            user_embeds.append(user_embeds_new)
            item_embeds.append(item_embeds_new)
        
        # 返回所有层的嵌入
        return user_embeds, item_embeds


class SocialGCN(nn.Module):
    """SocialGCN卷积模块"""
    def __init__(self, n_users, social_matrix, embed_dim, n_layers=2, dropout=0.1):
        super(SocialGCN, self).__init__()
        
        self.n_users = n_users
        self.n_layers = n_layers
        self.dropout = dropout
        
        # 归一化社交邻接矩阵
        if not sp.issparse(social_matrix):
            social_matrix = sp.csr_matrix(social_matrix)
        
        # 处理度为0的情况
        social_degrees = np.array(social_matrix.sum(axis=1)).flatten()
        social_degrees[social_degrees == 0] = 1
        
        # 计算归一化社交邻接矩阵
        social_diag_inv = sp.diags(1 / social_degrees)
        self.S = social_diag_inv @ social_matrix
        
        # 将稀疏矩阵转为PyTorch的稀疏格式
        self.S_indices = torch.LongTensor(np.vstack(self.S.nonzero()).astype(np.int64))
        self.S_values = torch.FloatTensor(self.S.data.astype(np.float32))
        
        # 社交变换权重
        self.W_s = nn.ParameterList([
            nn.Parameter(torch.Tensor(embed_dim, embed_dim))
            for _ in range(n_layers)
        ])
        
        # 初始化参数
        for w in self.W_s:
            nn.init.xavier_uniform_(w)
    
    def to(self, device):
        super(SocialGCN, self).to(device)
        self.S_indices = self.S_indices.to(device)
        self.S_values = self.S_values.to(device)
        return self
    
    def forward(self, user_embeds):
        # 初始嵌入
        user_embeds_s = [user_embeds]
        
        # 社交卷积层
        for l in range(self.n_layers):
            # 社交变换
            user_embeds_trans = user_embeds_s[-1] @ self.W_s[l]
            
            # 社交传播 - 修复设备不一致问题
            # 获取当前嵌入的设备，确保稀疏张量也在相同设备上
            device = user_embeds_trans.device
            
            # 创建稀疏张量并明确指定设备
            S_sparse = torch.sparse.FloatTensor(
                self.S_indices, 
                self.S_values,
                torch.Size([self.n_users, self.n_users])
            ).to(device)  # 确保稀疏张量在正确的设备上
            
            # 社交聚合
            user_embeds_social = torch.sparse.mm(S_sparse, user_embeds_trans)
            
            # 残差连接和dropout
            user_embeds_new = user_embeds_s[-1] + F.dropout(user_embeds_social, p=self.dropout, training=self.training)
            
            # 添加到嵌入列表
            user_embeds_s.append(user_embeds_new)
        
        # 返回所有层的嵌入
        return user_embeds_s


class MultiHeadAttention(nn.Module):
    """多头注意力模块"""
    def __init__(self, embed_dim, n_heads=2):
        super(MultiHeadAttention, self).__init__()
        
        self.embed_dim = embed_dim # 这是期望的输入和输出维度
        self.n_heads = n_heads
        
        # 确保内部维度可以被头数整除
        self.head_dim = embed_dim // n_heads
        if self.head_dim * n_heads != embed_dim:
             # 如果不能整除，调整内部处理维度，但最终输出仍需匹配embed_dim
            self.internal_dim = (embed_dim // n_heads + 1) * n_heads
            print(f"MultiHeadAttention 警告: 嵌入维度 {embed_dim} 不能被头数 {n_heads} 整除。内部调整为 {self.internal_dim}")
        else:
            self.internal_dim = embed_dim

        self.internal_head_dim = self.internal_dim // n_heads

        print(f"MultiHeadAttention初始化: 输入/输出维度={self.embed_dim}, 头数={n_heads}, 内部维度={self.internal_dim}, 内部头维度={self.internal_head_dim}")
        
        # Q, K, V 线性投影: 输入 embed_dim -> 输出 internal_dim
        self.W_Q = nn.Linear(self.embed_dim, self.internal_dim)
        self.W_K = nn.Linear(self.embed_dim, self.internal_dim)
        self.W_V = nn.Linear(self.embed_dim, self.internal_dim)
        
        # 输出投影: 输入 internal_dim -> 输出 embed_dim (匹配原始输入)
        self.W_O = nn.Linear(self.internal_dim, self.embed_dim)
    
    def forward(self, z):
        # z 的预期形状: [batch_size, seq_len, embed_dim]
        batch_size = z.size(0)
        seq_len = z.size(1) # 通常是1，因为我们 unsqueeze(1)
        
        # 检查输入维度是否匹配初始化时的 embed_dim
        if z.size(-1) != self.embed_dim:
             raise ValueError(f"MultiHeadAttention 输入维度错误: 期望 {self.embed_dim}, 收到 {z.size(-1)}")

        # 线性投影 Q, K, V
        Q = self.W_Q(z).view(batch_size, seq_len, self.n_heads, self.internal_head_dim).transpose(1, 2) # [batch, n_heads, seq_len, internal_head_dim]
        K = self.W_K(z).view(batch_size, seq_len, self.n_heads, self.internal_head_dim).transpose(1, 2) # [batch, n_heads, seq_len, internal_head_dim]
        V = self.W_V(z).view(batch_size, seq_len, self.n_heads, self.internal_head_dim).transpose(1, 2) # [batch, n_heads, seq_len, internal_head_dim]
        
        # 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.internal_head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights) # 处理 NaN
        
        attn_output = torch.matmul(attn_weights, V) # [batch, n_heads, seq_len, internal_head_dim]
        
        # 拼接多头输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.internal_dim) # [batch, seq_len, internal_dim]
        
        # 最终线性投影，恢复到原始 embed_dim
        output = self.W_O(attn_output) # [batch, seq_len, embed_dim]
        
        return output


class OutputMLP(nn.Module):
    """输出MLP模块"""
    def __init__(self, input_dim, output_dim):
        super(OutputMLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 隐藏层维度通常是输入或输出维度的一半或某个固定值
        # hidden_dim = (input_dim + output_dim) // 2 # 尝试这种策略
        hidden_dim = max(output_dim, input_dim // 2) # 或者取较大值
        # hidden_dim = output_dim * 2 # 或者基于输出维度
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim) 
        
        print(f"OutputMLP初始化: 输入维度={input_dim}, 隐藏维度={hidden_dim}, 输出维度={output_dim}")
        print(f"  - fc1 weight shape: {self.fc1.weight.shape}")
        print(f"  - fc2 weight shape: {self.fc2.weight.shape}")
        
    def forward(self, x):
        if x.size(-1) != self.input_dim:
            raise ValueError(f"OutputMLP 输入维度错误: 期望 {self.input_dim}, 收到 {x.size(-1)}")
            
        h1 = F.leaky_relu(self.layer_norm(self.fc1(x)))
        out = self.fc2(h1)
        return out


class LiteEIISRS(nn.Module):
    """Lite-EIISRS模型"""
    def __init__(self, n_users, n_items, embed_dim, interaction_matrix, social_matrix, 
                 feature_dim, n_layers_light=2, n_layers_social=2, n_heads=2, beta=0.2):
        super(LiteEIISRS, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim # 这是最终输出的用户/物品嵌入维度
        self.beta = beta
        self.feature_dim = feature_dim # 扩展后的特征维度 (e.g., 15)
        
        print(f"LiteEIISRS初始化 - 用户数:{n_users}, 物品数:{n_items}, 特征维度:{feature_dim}, 基础嵌入维度:{embed_dim}")
        
        # --- 子模块维度定义 ---
        # 1. MF/SVD/Stat 特征处理器的输出维度
        #    我们希望这三部分拼接后接近 embed_dim，所以每部分大约 embed_dim / 3
        #    同时，StatProcessor 需要一个输出维度
        stat_embed_dim = embed_dim // 4 # 让 Stat 输出 embed_dim/4
        mf_svd_embed_dim = embed_dim // 4 # MF 和 SVD 也输出 embed_dim/4
        self.actual_embed_dim = mf_svd_embed_dim * 2 + stat_embed_dim # 记录这部分的拼接维度
        # 让 LightGCN 和 SocialGCN 使用基础的 embed_dim
        lightgcn_embed_dim = embed_dim
        socialgcn_embed_dim = embed_dim # 通常与 LightGCN 一致

        print(f"  维度分配: MF/SVD={mf_svd_embed_dim}, Stat={stat_embed_dim}, LightGCN={lightgcn_embed_dim}, SocialGCN={socialgcn_embed_dim}")

        # --- 初始化子模块 ---
        self.mf = MFInitializer(n_users, n_items, mf_svd_embed_dim)
        self.svd = SVDInitializer(interaction_matrix, mf_svd_embed_dim)
        self.user_stat_processor = StatFeatureProcessor(feature_dim, stat_embed_dim)
        self.item_stat_processor = StatFeatureProcessor(feature_dim, stat_embed_dim)
        
        # LightGCN 使用指定的 lightgcn_embed_dim
        self.light_gcn = LightGCN(n_users, n_items, interaction_matrix, lightgcn_embed_dim, n_layers_light)
        # SocialGCN 也使用指定的 socialgcn_embed_dim
        self.social_gcn = SocialGCN(n_users, social_matrix, socialgcn_embed_dim, n_layers_social)
        
        # --- 计算拼接和 MLP 输入维度 ---
        # 用户拼接维度: MF + SVD + Stat + LightGCN + SocialGCN
        user_concat_dim = mf_svd_embed_dim + mf_svd_embed_dim + stat_embed_dim + lightgcn_embed_dim + socialgcn_embed_dim
        # 物品拼接维度: MF + SVD + Stat + LightGCN
        item_concat_dim = mf_svd_embed_dim + mf_svd_embed_dim + stat_embed_dim + lightgcn_embed_dim
        print(f"  拼接维度: 用户={user_concat_dim}, 物品={item_concat_dim}")

        # 多头注意力: 输入是拼接向量，输出维度也应与拼接维度相同，以便残差连接
        self.user_multi_head_attn = MultiHeadAttention(user_concat_dim, n_heads)
        self.item_multi_head_attn = MultiHeadAttention(item_concat_dim, n_heads)

        # 初始嵌入维度 (用于最终 MLP 输入和残差)
        user_initial_dim = mf_svd_embed_dim + mf_svd_embed_dim + stat_embed_dim
        item_initial_dim = mf_svd_embed_dim + mf_svd_embed_dim + stat_embed_dim
        print(f"  初始嵌入维度 (MF+SVD+Stat): {user_initial_dim}") # 用户和物品应该一样

        # 输出 MLP 输入维度: 初始嵌入 + 融合嵌入 (融合嵌入维度=拼接维度)
        user_mlp_in_dim = user_initial_dim + user_concat_dim
        item_mlp_in_dim = item_initial_dim + item_concat_dim
        print(f"  MLP 输入维度: 用户={user_mlp_in_dim}, 物品={item_mlp_in_dim}")
        
        # 输出 MLP: 输入是 (初始+融合)，输出是最终的 embed_dim
        self.user_output_mlp = OutputMLP(user_mlp_in_dim, self.embed_dim)
        self.item_output_mlp = OutputMLP(item_mlp_in_dim, self.embed_dim)
        
        # 最终残差连接需要维度匹配: OutputMLP 输出 (embed_dim) vs 初始嵌入 (user_initial_dim)
        # 如果 embed_dim != user_initial_dim，需要调整
        if self.embed_dim != user_initial_dim:
             print(f"  警告: 最终残差连接维度不匹配 ({self.embed_dim} vs {user_initial_dim})。添加调整层。")
             # 添加线性层将 initial 调整到 embed_dim
             self.initial_adjust = nn.Linear(user_initial_dim, self.embed_dim)
        else:
             self.initial_adjust = nn.Identity() # 维度匹配，无需调整

        print(f"  最终输出维度: {self.embed_dim}")
        print("-" * 30) # 分隔符

    def forward(self, user_ids, item_ids=None, user_features=None, item_features=None):
        device = user_ids.device
        
        # 1. 获取所有基础嵌入
        user_mf, item_mf = self.mf(user_ids, item_ids) # [B, mf_svd_dim]
        user_svd, item_svd = self.svd(user_ids, item_ids) # [B, mf_svd_dim]
        
        # 处理统计特征 (确保处理 None 情况)
        if user_features is not None:
            user_stat = self.user_stat_processor(user_features) # [B, stat_dim]
        else:
            # 需要知道 stat_embed_dim
            stat_dim = self.user_stat_processor.embed_dim
            user_stat = torch.zeros(user_ids.size(0), stat_dim, device=device)
            
        if item_features is not None and item_ids is not None:
            item_stat = self.item_stat_processor(item_features) # [B, stat_dim]
        elif item_ids is not None:
            stat_dim = self.item_stat_processor.embed_dim
            item_stat = torch.zeros(item_ids.size(0), stat_dim, device=device)
        else:
             item_stat = None # 如果没有 item_ids, item_stat 也是 None

        # LightGCN 传播 (获取所有用户/物品的最终嵌入)
        # 注意：light_gcn 输出的是列表，包含各层嵌入，我们需要最后一层
        all_user_embeds_light, all_item_embeds_light = self.light_gcn(None, None) 
        user_light = all_user_embeds_light[-1][user_ids] # [B, lightgcn_dim]
        item_light = all_item_embeds_light[-1][item_ids] if item_ids is not None else None # [B, lightgcn_dim]
        
        # SocialGCN 传播 (输入是 LightGCN 的最后一层用户嵌入)
        all_user_embeds_social = self.social_gcn(all_user_embeds_light[-1])
        user_social = all_user_embeds_social[-1][user_ids] # [B, socialgcn_dim]
        
        # 2. 拼接多源嵌入
        user_concat = torch.cat([user_mf, user_svd, user_stat, user_light, user_social], dim=-1)
        # print(f"Debug user_concat shape: {user_concat.shape}") # 应该匹配 user_concat_dim

        if item_ids is not None:
            item_concat = torch.cat([item_mf, item_svd, item_stat, item_light], dim=-1)
            # print(f"Debug item_concat shape: {item_concat.shape}") # 应该匹配 item_concat_dim
        
        # 3. 多头注意力 + 残差连接
        # unsqueeze(1) 在序列长度维度上增加一个维度，符合 MultiHeadAttention 的预期输入 [B, SeqLen, Dim]
        user_attn = self.user_multi_head_attn(user_concat.unsqueeze(1)).squeeze(1) # 输出 [B, user_concat_dim]
        user_fused = user_concat + user_attn # 残差连接，维度必须完全匹配

        if item_ids is not None:
            item_attn = self.item_multi_head_attn(item_concat.unsqueeze(1)).squeeze(1) # 输出 [B, item_concat_dim]
            item_fused = item_concat + item_attn # 残差连接
        
        # 4. 最终输出 MLP + 残差连接
        # 准备初始嵌入 (用于 MLP 输入和最终残差)
        user_initial = torch.cat([user_mf, user_svd, user_stat], dim=-1) # [B, user_initial_dim]
        
        # 准备 MLP 输入
        input_to_user_mlp = torch.cat([user_initial, user_fused], dim=-1) # [B, user_mlp_in_dim]
        # print(f"Debug input_to_user_mlp shape: {input_to_user_mlp.shape}")
        
        # 通过 MLP
        user_mlp_out = self.user_output_mlp(input_to_user_mlp) # [B, embed_dim]

        # 最终残差连接 (需要调整 initial 维度以匹配 mlp_out)
        user_final = user_mlp_out + self.beta * self.initial_adjust(user_initial) # [B, embed_dim]
        
        if item_ids is not None:
            item_initial = torch.cat([item_mf, item_svd, item_stat], dim=-1) # [B, item_initial_dim]
            input_to_item_mlp = torch.cat([item_initial, item_fused], dim=-1) # [B, item_mlp_in_dim]
            # print(f"Debug input_to_item_mlp shape: {input_to_item_mlp.shape}")
            item_mlp_out = self.item_output_mlp(input_to_item_mlp) # [B, embed_dim]
            item_final = item_mlp_out + self.beta * self.initial_adjust(item_initial) # [B, embed_dim]
            return user_final, item_final
        
        return user_final

    # predict 和 reg_loss 方法不变
    def predict(self, user_ids, item_ids, user_features=None, item_features=None):
        """预测用户对物品的评分"""
        # 确保在评估模式下调用 forward
        self.eval() 
        with torch.no_grad():
             user_emb, item_emb = self.forward(user_ids, item_ids, user_features, item_features)
        # 切换回训练模式（如果之后还需要训练）
        # self.train() 
        ratings = torch.sum(user_emb * item_emb, dim=1)
        return ratings
    
    def reg_loss(self):
        """正则化损失 - 目前只考虑MF部分"""
        # TODO: 可以考虑加入其他嵌入层的正则化
        return self.mf.reg_loss() 