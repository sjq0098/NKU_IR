import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import os

class SocialDataset(Dataset):
    def __init__(self, user_item_matrix, social_matrix, num_negatives=1):
        """
        初始化数据集
        Args:
            user_item_matrix: 用户-物品交互矩阵
            social_matrix: 社交关系矩阵
            num_negatives: 每个正样本对应的负样本数量
        """
        self.user_item_matrix = user_item_matrix
        self.social_matrix = social_matrix
        self.num_negatives = num_negatives
        self.num_users, self.num_items = user_item_matrix.shape
        
        # 获取所有正样本对
        self.pos_pairs = self._get_pos_pairs()
        
    def _get_pos_pairs(self):
        """获取所有正样本对"""
        pos_pairs = []
        for u in range(self.num_users):
            items = self.user_item_matrix[u].nonzero()[1]
            for i in items:
                pos_pairs.append((u, i))
        return pos_pairs
    
    def __len__(self):
        return len(self.pos_pairs)
    
    def __getitem__(self, idx):
        u, i = self.pos_pairs[idx]
        
        # 采样负样本
        neg_items = []
        for _ in range(self.num_negatives):
            while True:
                neg_i = np.random.randint(0, self.num_items)
                if neg_i not in self.user_item_matrix[u].nonzero()[1]:
                    neg_items.append(neg_i)
                    break
        
        # 获取用户的社交邻居，最多取前30个
        social_neighbors = self.social_matrix[u].nonzero()[1]
        if len(social_neighbors) > 30:
            social_neighbors = social_neighbors[:30]
        
        return {
            'user': u,
            'pos_item': i,
            'neg_items': neg_items,
            'social_neighbors': social_neighbors
        }

def collate_fn(batch):
    """自定义的collate函数，处理不同长度的社交邻居列表"""
    users = torch.tensor([item['user'] for item in batch])
    pos_items = torch.tensor([item['pos_item'] for item in batch])
    
    # 处理负样本
    neg_items_list = [item['neg_items'] for item in batch]
    neg_items = torch.tensor(neg_items_list)
    
    # 处理社交邻居 - 使用填充
    max_neighbors = max([len(item['social_neighbors']) for item in batch])
    social_neighbors_padded = []
    
    for item in batch:
        neighbors = item['social_neighbors']
        # 填充到最大长度
        if len(neighbors) < max_neighbors:
            padding = [-1] * (max_neighbors - len(neighbors))
            neighbors = np.concatenate([neighbors, padding])
        social_neighbors_padded.append(neighbors)
    
    social_neighbors = torch.tensor(social_neighbors_padded)
    
    return {
        'user': users,
        'pos_item': pos_items,
        'neg_items': neg_items,
        'social_neighbors': social_neighbors
    }

def load_data(dataset_name, data_path):
    """
    加载数据集
    Args:
        dataset_name: 数据集名称
        data_path: 数据路径
    Returns:
        user_item_matrix: 用户-物品交互矩阵
        social_matrix: 社交关系矩阵
    """
    # 加载用户-物品交互数据
    interactions = pd.read_csv(f"{data_path}/{dataset_name}/interactions.csv")
    user_item_matrix = csr_matrix((
        np.ones(len(interactions)),
        (interactions['user_id'], interactions['item_id'])
    ))
    
    # 加载社交关系数据
    social_relations = pd.read_csv(f"{data_path}/{dataset_name}/social.csv")
    social_matrix = csr_matrix((
        np.ones(len(social_relations)),
        (social_relations['user_id'], social_relations['friend_id'])
    ))
    
    return user_item_matrix, social_matrix

class LastFMDataset(Dataset):
    def __init__(self, config, is_train=True):
        self.config = config
        self.is_train = is_train
        
        # 加载交互数据
        interactions_path = os.path.join(config.DATA_DIR, config.INTERACTION_FILE)
        social_path = os.path.join(config.DATA_DIR, config.SOCIAL_FILE)
        
        # 如果数据文件不存在，生成示例数据
        if not os.path.exists(interactions_path) or not os.path.exists(social_path):
            self._generate_sample_data()
        
        # 读取数据
        self.interactions = pd.read_csv(interactions_path)
        self.social = pd.read_csv(social_path)
        
        # 获取用户和物品的数量
        self.num_users = self.interactions['user_id'].max() + 1
        self.num_items = self.interactions['item_id'].max() + 1
        
        # 构建用户-物品交互矩阵
        self.user_item_matrix = np.zeros((self.num_users, self.num_items))
        for _, row in self.interactions.iterrows():
            self.user_item_matrix[row['user_id'], row['item_id']] = 1
        
        # 构建社交关系矩阵
        self.social_matrix = np.zeros((self.num_users, self.num_users))
        for _, row in self.social.iterrows():
            self.social_matrix[row['user_id'], row['friend_id']] = 1
            self.social_matrix[row['friend_id'], row['user_id']] = 1
        
        # 划分训练集和测试集
        if is_train:
            self.user_item_matrix = self.user_item_matrix * 0.8  # 使用80%的数据作为训练集
        else:
            self.user_item_matrix = self.user_item_matrix * 0.2  # 使用20%的数据作为测试集
    
    def _generate_sample_data(self):
        """生成示例数据"""
        # 生成用户-物品交互数据
        num_users = 1000
        num_items = 2000
        num_interactions = 5000
        
        interactions = []
        for _ in range(num_interactions):
            user_id = np.random.randint(0, num_users)
            item_id = np.random.randint(0, num_items)
            interactions.append([user_id, item_id])
        
        interactions_df = pd.DataFrame(interactions, columns=['user_id', 'item_id'])
        interactions_df.to_csv(os.path.join(self.config.DATA_DIR, self.config.INTERACTION_FILE), index=False)
        
        # 生成社交关系数据
        num_social = 2000
        social = []
        for _ in range(num_social):
            user_id = np.random.randint(0, num_users)
            friend_id = np.random.randint(0, num_users)
            if user_id != friend_id:
                social.append([user_id, friend_id])
        
        social_df = pd.DataFrame(social, columns=['user_id', 'friend_id'])
        social_df.to_csv(os.path.join(self.config.DATA_DIR, self.config.SOCIAL_FILE), index=False)
    
    def __len__(self):
        return self.num_users
    
    def __getitem__(self, idx):
        # 获取用户的交互物品
        pos_items = np.where(self.user_item_matrix[idx] == 1)[0]
        if len(pos_items) == 0:
            pos_items = [np.random.randint(0, self.num_items)]
        
        # 随机选择一个正样本
        pos_item = np.random.choice(pos_items)
        
        # 随机选择一个负样本
        neg_items = np.where(self.user_item_matrix[idx] == 0)[0]
        neg_item = np.random.choice(neg_items)
        
        # 获取用户的社交邻居
        neighbors = np.where(self.social_matrix[idx] == 1)[0]
        if len(neighbors) == 0:
            neighbors = [idx]  # 如果没有邻居，使用自己
        
        return {
            'user': idx,
            'pos_item': pos_item,
            'neg_items': torch.tensor([neg_item]),
            'neighbors': torch.tensor(neighbors)
        }
    
    def collate_fn(self, batch):
        """处理批次数据"""
        users = torch.tensor([item['user'] for item in batch])
        pos_items = torch.tensor([item['pos_item'] for item in batch])
        neg_items = torch.stack([item['neg_items'] for item in batch])
        
        # 处理不同长度的邻居列表
        max_neighbors = max(len(item['neighbors']) for item in batch)
        neighbors = torch.zeros(len(batch), max_neighbors, dtype=torch.long)
        for i, item in enumerate(batch):
            neighbors[i, :len(item['neighbors'])] = item['neighbors']
        
        return {
            'user': users,
            'pos_item': pos_items,
            'neg_items': neg_items,
            'neighbors': neighbors
        } 