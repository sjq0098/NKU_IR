import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Optional, List
import scipy.sparse as sp
import random

class SocialDataset(Dataset):
    def __init__(self, 
                 user_item_matrix: sp.spmatrix,
                 social_matrix: Optional[sp.spmatrix] = None,
                 user_features: Optional[np.ndarray] = None,
                 item_features: Optional[np.ndarray] = None,
                 num_negatives: int = 4,
                 mode: str = 'train'):
        """
        社交推荐数据集
        
        Args:
            user_item_matrix: 用户-物品交互矩阵
            social_matrix: 社交关系矩阵 (可选)
            user_features: 用户特征 (可选)
            item_features: 物品特征 (可选)
            num_negatives: 每个正样本对应的负样本数量
            mode: 数据集模式 ('train', 'val', 'test')
        """
        self.user_item_matrix = user_item_matrix
        self.social_matrix = social_matrix
        self.user_features = user_features
        self.item_features = item_features
        self.num_negatives = num_negatives
        self.mode = mode
        
        # 获取非零元素的位置
        self.user_indices, self.item_indices = user_item_matrix.nonzero()
        
        # 为每个用户生成物品字典
        self.user_items_dict = {}
        for u, i in zip(self.user_indices, self.item_indices):
            if u not in self.user_items_dict:
                self.user_items_dict[u] = []
            self.user_items_dict[u].append(i)
            
        # 所有物品的集合
        self.all_items = set(range(user_item_matrix.shape[1]))
        
        # 记录数据集大小
        self.num_users = user_item_matrix.shape[0]
        self.num_items = user_item_matrix.shape[1]
        
    def __len__(self) -> int:
        return len(self.user_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        user_idx = self.user_indices[idx]
        pos_item_idx = self.item_indices[idx]
        
        # 为当前用户生成负样本
        neg_item_indices = self._sample_negative_items(user_idx, self.num_negatives)
        
        # 创建样本字典
        sample = {
            'user_idx': torch.LongTensor([user_idx]),
            'pos_item_idx': torch.LongTensor([pos_item_idx]),
            'neg_item_idx': torch.LongTensor(neg_item_indices)
        }
        
        # 添加特征（如果有）
        if self.user_features is not None:
            sample['user_features'] = torch.FloatTensor(self.user_features[user_idx])
        if self.item_features is not None:
            sample['pos_item_features'] = torch.FloatTensor(self.item_features[pos_item_idx])
            if len(neg_item_indices) > 0:
                sample['neg_item_features'] = torch.FloatTensor(self.item_features[neg_item_indices])
            
        return sample
        
    def _sample_negative_items(self, user_idx: int, num_samples: int) -> List[int]:
        """为用户采样负样本物品"""
        # 获取用户已交互的物品
        pos_items = set(self.user_items_dict.get(user_idx, []))
        
        # 采样负样本
        neg_items = []
        while len(neg_items) < num_samples:
            # 随机选择一个物品
            neg_item = random.randint(0, self.num_items - 1)
            
            # 确保该物品不在用户的正样本集中
            if neg_item not in pos_items and neg_item not in neg_items:
                neg_items.append(neg_item)
                
        return neg_items

def create_data_loaders(
    user_item_matrix: sp.spmatrix,
    social_matrix: Optional[sp.spmatrix] = None,
    user_features: Optional[np.ndarray] = None,
    item_features: Optional[np.ndarray] = None,
    batch_size: int = 256,
    num_negatives: int = 4,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证和测试数据加载器
    """
    # 设置随机种子
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    
    # 获取交互数量
    num_interactions = user_item_matrix.nnz
    
    # 生成随机索引
    all_indices = np.random.permutation(num_interactions)
    
    # 划分训练、验证和测试集
    train_size = int(train_ratio * num_interactions)
    val_size = int(val_ratio * num_interactions)
    
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:train_size + val_size]
    test_indices = all_indices[train_size + val_size:]
    
    # 创建数据集
    train_dataset = SocialDataset(
        user_item_matrix=user_item_matrix,
        social_matrix=social_matrix,
        user_features=user_features,
        item_features=item_features,
        num_negatives=num_negatives,
        mode='train'
    )
    
    val_dataset = SocialDataset(
        user_item_matrix=user_item_matrix,
        social_matrix=social_matrix,
        user_features=user_features,
        item_features=item_features,
        num_negatives=num_negatives,
        mode='val'
    )
    
    test_dataset = SocialDataset(
        user_item_matrix=user_item_matrix,
        social_matrix=social_matrix,
        user_features=user_features,
        item_features=item_features,
        num_negatives=num_negatives,
        mode='test'
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices),
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices),
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(test_indices),
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader 