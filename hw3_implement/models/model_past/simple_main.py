import os
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import logging
from tqdm import tqdm
import random
from typing import Dict, List, Optional, Tuple
from social_recommender_simplified import SocialRecommenderSimplified
import torch.nn as nn
import matplotlib.pyplot as plt
import json
from pathlib import Path
from enhanced_social_recommender import EnhancedSocialRecommender

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 超参数
config = {
    # 模型参数
    "embed_dim": 64,
    "n_layers": 2,
    "dropout": 0.1,
    "temperature": 0.2,
    
    # 训练参数
    "batch_size": 1024,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "num_epochs": 100,
    "early_stopping_patience": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 0,  # Windows下建议设为0
    
    # 数据参数
    "data_path": "data",
    "dataset": "hetrec2011-lastfm-2k",
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "random_seed": 42,
    "num_negatives": 4
}

# TensorBoard日志 - 确保在config定义之后初始化
log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(log_dir)
logger.info(f"TensorBoard日志目录: {log_dir}")

# 添加说明信息到TensorBoard
writer.add_text('实验说明', f'LastFM数据集社交推荐模型训练 - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
writer.add_text('模型配置', str(config))

# 确保目录存在
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# 设置随机种子
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 简化的数据集
class SimpleSocialDataset(Dataset):
    def __init__(self, user_item_pairs, num_users, num_items, num_negatives=4):
        self.user_item_pairs = user_item_pairs  # 用户-物品交互对
        self.num_users = num_users
        self.num_items = num_items
        self.num_negatives = num_negatives
        
        # 创建用户的物品集合
        self.user_items = {}
        for user, item in self.user_item_pairs:
            if user not in self.user_items:
                self.user_items[user] = set()
            self.user_items[user].add(item)
    
    def __len__(self):
        return len(self.user_item_pairs)
    
    def __getitem__(self, idx):
        user, pos_item = self.user_item_pairs[idx]
        
        # 采样负样本
        neg_items = []
        while len(neg_items) < self.num_negatives:
            neg_item = random.randint(0, self.num_items - 1)
            if neg_item not in self.user_items.get(user, set()) and neg_item not in neg_items:
                neg_items.append(neg_item)
        
        return {
            'user_idx': torch.LongTensor([user]),
            'pos_item_idx': torch.LongTensor([pos_item]),
            'neg_item_idx': torch.LongTensor(neg_items).unsqueeze(0)  # 形状为 [1, num_negatives]
        }

# 数据加载函数
def load_data(data_path: str, dataset: str) -> Tuple:
    """加载LastFM数据集"""
    # 设置路径
    data_dir = os.path.join(data_path, dataset)
    user_artists_file = os.path.join(data_dir, 'user_artists.dat')
    user_friends_file = os.path.join(data_dir, 'user_friends.dat')
    
    logger.info(f"加载数据文件: {user_artists_file}")
    logger.info(f"加载数据文件: {user_friends_file}")
    
    # 加载用户-艺术家交互数据
    user_artists_df = pd.read_csv(user_artists_file, sep='\t')
    
    # 加载用户社交关系数据
    user_friends_df = pd.read_csv(user_friends_file, sep='\t')
    
    logger.info(f"用户-艺术家数据: {user_artists_df.shape}")
    logger.info(f"用户-朋友数据: {user_friends_df.shape}")
    
    # 获取唯一用户和艺术家
    unique_users = sorted(set(user_artists_df['userID'].unique()).union(
                         set(user_friends_df['userID'].unique())))
    unique_artists = sorted(user_artists_df['artistID'].unique())
    
    # ID映射 (确保从0开始)
    user_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
    artist_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_artists)}
    
    # 应用ID映射
    user_artists_mapped = user_artists_df.copy()
    user_artists_mapped['userID'] = user_artists_mapped['userID'].map(user_id_map)
    user_artists_mapped['artistID'] = user_artists_mapped['artistID'].map(artist_id_map)
    
    user_friends_mapped = user_friends_df.copy()
    user_friends_mapped['userID'] = user_friends_mapped['userID'].map(user_id_map)
    user_friends_mapped['friendID'] = user_friends_mapped['friendID'].apply(
        lambda x: user_id_map.get(x, -1)
    )
    # 过滤掉不在用户集中的好友
    user_friends_mapped = user_friends_mapped[user_friends_mapped['friendID'] != -1]
    
    # 创建用户-物品交互矩阵 (稀疏)
    num_users = len(unique_users)
    num_items = len(unique_artists)
    logger.info(f"用户数: {num_users}, 物品数: {num_items}")
    
    # 使用权重作为交互强度
    rows = user_artists_mapped['userID'].values
    cols = user_artists_mapped['artistID'].values
    data = user_artists_mapped['weight'].values
    
    # 归一化权重 (可选)
    max_weight = np.max(data)
    data = data / max_weight
    
    # 创建稀疏矩阵
    user_item_matrix = csr_matrix((data, (rows, cols)), shape=(num_users, num_items))
    
    # 创建社交关系矩阵 (稀疏)
    rows = user_friends_mapped['userID'].values
    cols = user_friends_mapped['friendID'].values
    data = np.ones_like(rows)
    
    social_matrix = csr_matrix((data, (rows, cols)), shape=(num_users, num_users))
    
    # 对称化社交矩阵(确保双向关系)
    social_matrix = social_matrix + social_matrix.T
    social_matrix.data = np.ones_like(social_matrix.data)  # 将值重置为1
    
    # 获取交互对
    interactions = list(zip(user_artists_mapped['userID'].values, 
                         user_artists_mapped['artistID'].values))
    
    return user_item_matrix, social_matrix, interactions, num_users, num_items

# 将scipy稀疏矩阵转为pytorch稀疏张量
def sparse_mx_to_torch_sparse(sparse_mx):
    """将 scipy 稀疏矩阵转换为 torch 稀疏张量"""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data.astype(np.float32))
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# 生成模型所需的邻接矩阵
def preprocess_adj(adj):
    """预处理邻接矩阵: 添加自连接和归一化"""
    if adj.shape[0] == adj.shape[1]:  # 方形矩阵
        # 添加自连接
        adj = adj + sp.eye(adj.shape[0])
    
    # 对行进行归一化
    rowsum = np.array(adj.sum(axis=1)).flatten()
    rowsum[rowsum == 0] = 1  # 避免除以0
    d_inv = 1.0 / rowsum
    d_mat_inv = sp.diags(d_inv)
    adj_normalized = d_mat_inv.dot(adj)
    
    return adj_normalized

# 训练一个epoch
def train_epoch(model, train_loader, optimizer, device, ui_adj, iu_adj, social_adj=None):
    model.train()
    total_loss = 0
    total_bpr_loss = 0
    total_reg_loss = 0
    total_social_loss = 0
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="训练")):
        # 将数据移到设备上
        user_idx = batch['user_idx'].to(device)
        pos_item_idx = batch['pos_item_idx'].to(device)
        neg_item_idx = batch['neg_item_idx'].to(device)  # 形状为 [batch_size, 1, num_negatives]
        
        # 调整维度
        user_idx = user_idx.squeeze()  # [batch_size]
        pos_item_idx = pos_item_idx.squeeze()  # [batch_size]
        neg_item_idx = neg_item_idx.squeeze(1)  # [batch_size, num_negatives]
        
        # 打印批次数据形状（仅第一个批次）
        if batch_idx == 0:
            logger.info(f"批次数据形状: users={user_idx.shape}, pos_items={pos_item_idx.shape}, neg_items={neg_item_idx.shape}")
        
        # 前向传播
        try:
            output = model(
                users=user_idx,
                pos_items=pos_item_idx,
                neg_items=neg_item_idx,
                ui_adj_mat={
                    'user_item': ui_adj,
                    'item_user': iu_adj
                },
                social_adj_mat=social_adj
            )
            
            # 计算损失
            loss_dict = model.calculate_loss(output)
            loss = loss_dict['total_loss']
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 累加损失
            total_loss += loss.item()
            total_bpr_loss += loss_dict['bpr_loss'].item()
            total_reg_loss += loss_dict['reg_loss'].item()
            if 'social_loss' in loss_dict:
                total_social_loss += loss_dict['social_loss'].item()
                
            # 打印第一个批次的损失（调试用）
            if batch_idx == 0:
                logger.info(f"第一个批次损失: total={loss.item():.4f}, bpr={loss_dict['bpr_loss'].item():.4f}")
                
        except Exception as e:
            logger.error(f"批次 {batch_idx} 处理失败: {e}")
            logger.error(f"错误详情: {str(e)}")
            # 跳过此批次
            continue
    
    # 计算平均损失
    num_batches = len(train_loader)
    return {
        'train_loss': total_loss / num_batches,
        'train_bpr_loss': total_bpr_loss / num_batches,
        'train_reg_loss': total_reg_loss / num_batches,
        'train_social_loss': total_social_loss / num_batches
    }

# 验证
def validate(model, val_loader, device, ui_adj, iu_adj, social_adj=None):
    model.eval()
    total_loss = 0
    total_bpr_loss = 0
    total_reg_loss = 0
    total_social_loss = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="验证")):
            # 将数据移到设备上
            user_idx = batch['user_idx'].to(device)
            pos_item_idx = batch['pos_item_idx'].to(device)
            neg_item_idx = batch['neg_item_idx'].to(device)  # 形状为 [batch_size, 1, num_negatives]
            
            # 调整维度
            user_idx = user_idx.squeeze()  # [batch_size]
            pos_item_idx = pos_item_idx.squeeze()  # [batch_size]
            neg_item_idx = neg_item_idx.squeeze(1)  # [batch_size, num_negatives]
            
            try:
                # 前向传播
                output = model(
                    users=user_idx,
                    pos_items=pos_item_idx,
                    neg_items=neg_item_idx,
                    ui_adj_mat={
                        'user_item': ui_adj,
                        'item_user': iu_adj
                    },
                    social_adj_mat=social_adj
                )
                
                # 计算损失
                loss_dict = model.calculate_loss(output)
                
                # 累加损失
                total_loss += loss_dict['total_loss'].item()
                total_bpr_loss += loss_dict['bpr_loss'].item()
                total_reg_loss += loss_dict['reg_loss'].item()
                if 'social_loss' in loss_dict:
                    total_social_loss += loss_dict['social_loss'].item()
            except Exception as e:
                logger.error(f"验证批次 {batch_idx} 处理失败: {e}")
                logger.error(f"错误详情: {str(e)}")
                # 跳过此批次
                continue
    
    # 计算平均损失
    num_batches = len(val_loader)
    return {
        'val_loss': total_loss / num_batches,
        'val_bpr_loss': total_bpr_loss / num_batches,
        'val_reg_loss': total_reg_loss / num_batches,
        'val_social_loss': total_social_loss / num_batches
    }

# 测试指标计算
def compute_metrics(model, test_loader, device, ui_adj, iu_adj, social_adj=None, k_list=[5, 10, 20], max_items=None):
    """计算推荐指标：HR@k, NDCG@k"""
    model.eval()
    
    metrics = {f'HR@{k}': [] for k in k_list}
    metrics.update({f'NDCG@{k}': [] for k in k_list})
    
    # 缓存用户的已交互物品，用于过滤
    if hasattr(test_loader.dataset, 'user_items'):
        user_items_dict = test_loader.dataset.user_items
    else:
        user_items_dict = {}  # 如果数据集没有这个属性，使用空字典
    
    # 限制计算物品数量以提高效率
    if max_items is None:
        max_items = model.num_items
    else:
        max_items = min(max_items, model.num_items)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="计算推荐指标")):
            # 将数据移到设备上
            user_idx = batch['user_idx'].to(device)
            pos_item_idx = batch['pos_item_idx'].to(device)
            
            # 调整维度
            user_idx = user_idx.squeeze()  # [batch_size]
            pos_item_idx = pos_item_idx.squeeze()  # [batch_size]
            
            try:
                # 为每个用户获取所有物品的预测得分
                output = model(
                    users=user_idx,
                    pos_items=None,  # 预测模式
                    neg_items=None,
                    ui_adj_mat={
                        'user_item': ui_adj,
                        'item_user': iu_adj
                    },
                    social_adj_mat=social_adj
                )
                
                scores = output['scores']  # [batch_size, num_items]
                
                # 如果物品数量太大，只考虑前max_items个
                if scores.shape[1] > max_items:
                    scores = scores[:, :max_items]
                
                # 对于每个用户，计算HR@k和NDCG@k
                for i, user in enumerate(user_idx):
                    user_id = user.item()
                    # 获取正样本物品
                    pos_item = pos_item_idx[i].item()
                    
                    # 获取用户的推荐物品排名
                    user_scores = scores[i]
                    
                    # 过滤掉训练集中的物品（如果可能）
                    if user_id in user_items_dict:
                        # 创建掩码，将训练集中的物品分数设为负无穷
                        mask = torch.ones_like(user_scores, dtype=torch.bool)
                        train_items = list(user_items_dict[user_id])
                        # 只过滤在当前评分范围内的物品
                        valid_train_items = [item for item in train_items if item < user_scores.shape[0]]
                        if valid_train_items and pos_item not in valid_train_items:
                            mask[valid_train_items] = False
                            # 保留测试物品的分数
                            if pos_item < user_scores.shape[0]:
                                mask[pos_item] = True
                            # 应用掩码
                            user_scores = user_scores.masked_fill(~mask, float('-inf'))
                    
                    # 如果正样本超出了考虑的物品范围，跳过这个用户
                    if pos_item >= max_items:
                        continue
                        
                    # 获取前k个推荐物品的索引
                    _, indices = torch.sort(user_scores, descending=True)
                    indices = indices.cpu().numpy()
                    
                    # 计算HR@k和NDCG@k
                    for k in k_list:
                        # 考虑到实际物品数量可能小于k
                        top_k = min(k, len(indices))
                        top_k_items = indices[:top_k]
                        
                        # HR@k
                        hr_k = 1.0 if pos_item in top_k_items else 0.0
                        metrics[f'HR@{k}'].append(hr_k)
                        
                        # NDCG@k
                        if hr_k == 1.0:
                            rank = np.where(top_k_items == pos_item)[0][0] + 1
                            ndcg_k = 1.0 / np.log2(rank + 1)
                        else:
                            ndcg_k = 0.0
                        metrics[f'NDCG@{k}'].append(ndcg_k)
                        
            except Exception as e:
                logger.error(f"评估批次 {batch_idx} 处理失败: {e}")
                logger.error(f"错误详情: {str(e)}")
                continue
    
    # 计算平均指标
    for k in metrics:
        if len(metrics[k]) > 0:
            metrics[k] = np.mean(metrics[k])
        else:
            metrics[k] = 0.0
            logger.warning(f"指标 {k} 没有有效样本，设置为0")
        
    return metrics

# 指标可视化函数
def visualize_metrics(metrics_history, save_dir="metrics_plots"):
    """将训练过程中的指标绘制成图表
    
    Args:
        metrics_history: 包含每个epoch指标的字典
        save_dir: 保存图表的目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 整理指标数据
    epochs = list(range(1, len(metrics_history['train_loss']) + 1))
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, metrics_history['train_loss'], 'b-', label='训练损失')
    plt.plot(epochs, metrics_history['val_loss'], 'r-', label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'), dpi=300)
    plt.close()
    
    # 绘制HR@K曲线
    plt.figure(figsize=(12, 6))
    for k in [5, 10, 20]:
        key = f'HR@{k}'
        if key in metrics_history:
            plt.plot(epochs, metrics_history[key], label=key)
    plt.title('HR@K指标')
    plt.xlabel('Epoch')
    plt.ylabel('HR@K')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'hr_curves.png'), dpi=300)
    plt.close()
    
    # 绘制NDCG@K曲线
    plt.figure(figsize=(12, 6))
    for k in [5, 10, 20]:
        key = f'NDCG@{k}'
        if key in metrics_history:
            plt.plot(epochs, metrics_history[key], label=key)
    plt.title('NDCG@K指标')
    plt.xlabel('Epoch')
    plt.ylabel('NDCG@K')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'ndcg_curves.png'), dpi=300)
    plt.close()
    
    # 保存原始数据为JSON
    with open(os.path.join(save_dir, 'metrics_history.json'), 'w') as f:
        # 将numpy数组转换为列表以便序列化
        serializable_metrics = {}
        for key, value in metrics_history.items():
            serializable_metrics[key] = [float(x) for x in value]
        json.dump(serializable_metrics, f, indent=4)
    
    logger.info(f"指标可视化图表已保存到 {save_dir} 目录")

# 主函数
def main():
    set_seed(config['random_seed'])
    device = torch.device(config['device'])
    
    # 记录训练过程中的指标
    metrics_history = {
        'train_loss': [],
        'val_loss': [],
    }
    # 为每个K值添加HR和NDCG的记录
    for k in [5, 10, 20]:
        metrics_history[f'HR@{k}'] = []
        metrics_history[f'NDCG@{k}'] = []
    
    # 加载数据
    logger.info("加载数据...")
    user_item_matrix, social_matrix, interactions, num_users, num_items = load_data(
        config['data_path'], config['dataset']
    )
    
    # 数据拆分
    np.random.shuffle(interactions)
    num_interactions = len(interactions)
    train_size = int(config['train_ratio'] * num_interactions)
    val_size = int(config['val_ratio'] * num_interactions)
    
    train_interactions = interactions[:train_size]
    val_interactions = interactions[train_size:train_size + val_size]
    test_interactions = interactions[train_size + val_size:]
    
    logger.info(f"训练集: {len(train_interactions)}, 验证集: {len(val_interactions)}, 测试集: {len(test_interactions)}")
    
    # 创建数据集和数据加载器
    train_dataset = SimpleSocialDataset(train_interactions, num_users, num_items, config['num_negatives'])
    val_dataset = SimpleSocialDataset(val_interactions, num_users, num_items, config['num_negatives'])
    test_dataset = SimpleSocialDataset(test_interactions, num_users, num_items, config['num_negatives'])
    
    # 降低批次大小，减少内存占用
    config['batch_size'] = min(config['batch_size'], 64)
    logger.info(f"使用批次大小: {config['batch_size']}")
    
    # 创建评估数据加载器（小批量用于快速评估）
    eval_batch_size = min(config['batch_size'], 32)
    logger.info(f"评估批次大小: {eval_batch_size}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers']
    )
    
    # 用于评估的小批量加载器
    eval_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers']
    )
    
    # 预处理邻接矩阵
    logger.info("预处理邻接矩阵...")
    user_item_adj = preprocess_adj(user_item_matrix)
    item_user_adj = preprocess_adj(user_item_matrix.T)
    social_adj = preprocess_adj(social_matrix)
    
    # 将稀疏矩阵转为PyTorch张量
    ui_adj = sparse_mx_to_torch_sparse(user_item_adj).to(device)
    iu_adj = sparse_mx_to_torch_sparse(item_user_adj).to(device)
    social_adj_tensor = sparse_mx_to_torch_sparse(social_adj).to(device)
    
    # 创建模型
    logger.info("初始化模型...")
    
    # 首先定义一个简单模型类，作为备用
    class SimpleModel(nn.Module):
        def __init__(self, num_users, num_items, embed_dim=64):
            super().__init__()
            self.user_embedding = nn.Parameter(torch.randn(num_users, embed_dim))
            self.item_embedding = nn.Parameter(torch.randn(num_items, embed_dim))
            nn.init.xavier_uniform_(self.user_embedding)
            nn.init.xavier_uniform_(self.item_embedding)
            self.num_users = num_users
            self.num_items = num_items
            
        def forward(self, users, pos_items, neg_items, ui_adj_mat, social_adj_mat=None):
            batch_user_emb = self.user_embedding[users]
            
            # 预测模式
            if neg_items is None:
                scores = torch.matmul(batch_user_emb, self.item_embedding.t())
                return {'scores': scores}
                
            # 训练模式
            pos_item_emb = self.item_embedding[pos_items]
            neg_item_emb = self.item_embedding[neg_items]
            
            pos_scores = torch.sum(batch_user_emb * pos_item_emb, dim=-1)
            neg_scores = torch.sum(batch_user_emb * neg_item_emb, dim=-1)
            
            return {
                'pos_scores': pos_scores,
                'neg_scores': neg_scores,
                'user_emb': batch_user_emb,
                'pos_item_emb': pos_item_emb,
                'neg_item_emb': neg_item_emb,
                'ui_emb': batch_user_emb,  # 占位符
                'social_emb': batch_user_emb,  # 占位符
                'final_user_emb': batch_user_emb
            }
        
        def calculate_loss(self, batch_output):
            if 'scores' in batch_output:
                return {'scores': batch_output['scores']}
                
            pos_scores = batch_output['pos_scores']
            neg_scores = batch_output['neg_scores']
            
            bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
            reg_loss = torch.tensor(0.0, device=pos_scores.device)
            social_loss = torch.tensor(0.0, device=pos_scores.device)
            
            return {
                'total_loss': bpr_loss,
                'bpr_loss': bpr_loss,
                'reg_loss': reg_loss,
                'social_loss': social_loss
            }
    
    # 尝试创建增强模型
    try:
        logger.info("尝试创建增强版社交推荐模型...")
        model = EnhancedSocialRecommender(
            num_users=num_users,
            num_items=num_items,
            embed_dim=config['embed_dim'],
            interaction_matrix=user_item_matrix,  
            social_matrix=social_matrix,
            n_layers=config['n_layers'],
            lightgcn_layers=2,
            social_layers=2,
            num_heads=4,
            temperature=config['temperature'],
            dropout=config['dropout']
        ).to(device)
        
        # 执行前向传播检查是否能正常工作
        logger.info("测试模型前向传播...")
        dummy_users = torch.LongTensor([0]).to(device)
        dummy_items = torch.LongTensor([0]).to(device)
        dummy_negs = torch.LongTensor([[1]]).to(device)
        
        with torch.no_grad():
            _ = model(
                users=dummy_users,
                pos_items=dummy_items,
                neg_items=dummy_negs,
                ui_adj_mat={'user_item': ui_adj, 'item_user': iu_adj},
                social_adj_mat=social_adj_tensor
            )
        logger.info("增强版模型前向传播测试通过!")
        
    except Exception as e:
        logger.error(f"增强版模型创建或测试失败: {e}")
        
        try:
            # 尝试创建简化版社交推荐模型
            logger.info("尝试创建简化版社交推荐模型...")
            model = SocialRecommenderSimplified(
                num_users=num_users,
                num_items=num_items,
                embed_dim=config['embed_dim'],
                interaction_matrix=user_item_matrix,  
                social_matrix=social_matrix,
                n_layers=config['n_layers'],
                dropout=config['dropout'],
                temperature=config['temperature']
            ).to(device)
            
            # 测试简化版模型
            logger.info("测试简化版模型前向传播...")
            dummy_users = torch.LongTensor([0]).to(device)
            dummy_items = torch.LongTensor([0]).to(device)
            dummy_negs = torch.LongTensor([[1]]).to(device)
            
            with torch.no_grad():
                _ = model(
                    users=dummy_users,
                    pos_items=dummy_items,
                    neg_items=dummy_negs,
                    ui_adj_mat={'user_item': ui_adj, 'item_user': iu_adj},
                    social_adj_mat=social_adj_tensor
                )
            logger.info("简化版模型前向传播测试通过!")
            
        except Exception as e:
            logger.error(f"简化版模型也失败了: {e}")
            logger.info("降级至基础矩阵分解模型...")
            
            # 创建超简化版模型（基础矩阵分解）
            model = SimpleModel(
                num_users=num_users,
                num_items=num_items,
                embed_dim=config['embed_dim']
            ).to(device)
            logger.info("使用基础矩阵分解模型")
    
    # 打印模型参数
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型总参数数量: {total_params:,}")
    
    # 优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # 学习率调度器（可选）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 定义在训练过程中要计算的K值
    k_list = [5, 10, 20]
    
    # 训练
    logger.info("开始训练...")
    best_val_loss = float('inf')
    best_hr10 = 0.0  # 记录最佳HR@10
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        # 训练一个epoch
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, ui_adj, iu_adj, social_adj_tensor
        )
        
        # 验证
        val_metrics = validate(
            model, val_loader, device, ui_adj, iu_adj, social_adj_tensor
        )
        
        # 计算推荐指标（在验证集的一个子集上）
        logger.info("计算当前epoch的推荐指标...")
        rec_metrics = compute_metrics(
            model, eval_loader, device, ui_adj, iu_adj, social_adj_tensor, k_list=k_list
        )
        
        # 学习率调度
        scheduler.step(val_metrics['val_loss'])
        
        # 记录指标到TensorBoard
        for name, value in train_metrics.items():
            writer.add_scalar(f'train/{name}', value, epoch)
        for name, value in val_metrics.items():
            writer.add_scalar(f'val/{name}', value, epoch)
        for name, value in rec_metrics.items():
            writer.add_scalar(f'metrics/{name}', value, epoch)
        
        # 保存指标历史
        metrics_history['train_loss'].append(train_metrics['train_loss'])
        metrics_history['val_loss'].append(val_metrics['val_loss'])
        for k in k_list:
            metrics_history[f'HR@{k}'].append(rec_metrics[f'HR@{k}'])
            metrics_history[f'NDCG@{k}'].append(rec_metrics[f'NDCG@{k}'])
        
        # 打印进度
        logger.info(f"周期 {epoch+1}/{config['num_epochs']}")
        logger.info(f"训练损失: {train_metrics['train_loss']:.4f}")
        logger.info(f"验证损失: {val_metrics['val_loss']:.4f}")
        
        # 打印推荐指标
        for k in k_list:
            logger.info(f"HR@{k}: {rec_metrics[f'HR@{k}']:.4f}, NDCG@{k}: {rec_metrics[f'NDCG@{k}']:.4f}")
        
        # 早停：使用HR@10作为主要指标
        current_hr10 = rec_metrics['HR@10']
        if current_hr10 > best_hr10:
            best_hr10 = current_hr10
            best_epoch = epoch
            patience_counter = 0
            
            # 保存最佳模型
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            logger.info(f"保存最佳模型, 周期 {epoch+1}, HR@10: {best_hr10:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                logger.info(f"早停激活, 最佳周期: {best_epoch+1}, 最佳HR@10: {best_hr10:.4f}")
                break
    
    # 绘制训练过程中的指标变化曲线
    visualize_metrics(metrics_history)
    
    # 加载最佳模型进行测试
    logger.info("加载最佳模型进行测试...")
    model.load_state_dict(torch.load('checkpoints/best_model.pth'))
    
    # 测试评估
    test_metrics = compute_metrics(
        model, test_loader, device, ui_adj, iu_adj, social_adj_tensor, k_list=k_list
    )
    
    # 输出测试指标
    logger.info("测试评估结果:")
    for metric, value in test_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
        writer.add_scalar(f'test/{metric}', value, 0)
    
    # 保存最终实验结果
    results = {
        'best_epoch': best_epoch + 1,
        'best_hr10': best_hr10,
        'test_metrics': test_metrics
    }
    
    # 保存实验结果为JSON
    with open(os.path.join(log_dir, 'final_results.json'), 'w') as f:
        # 将numpy值转换为Python原始类型
        for k, v in results['test_metrics'].items():
            results['test_metrics'][k] = float(v)
        json.dump(results, f, indent=4)
    
    logger.info("训练完成!")
    writer.close()
    
    # 显示最终测试结果表格
    logger.info("\n最终测试结果:")
    logger.info("-" * 50)
    logger.info(f"{'指标':<10} | {'值':>10}")
    logger.info("-" * 50)
    for k, v in test_metrics.items():
        logger.info(f"{k:<10} | {v:>10.4f}")
    logger.info("-" * 50)

if __name__ == "__main__":
    main() 