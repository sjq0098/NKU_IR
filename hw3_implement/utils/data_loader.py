import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Tuple, Dict, Optional

def load_lastfm_data(data_path: str) -> Tuple[sp.csr_matrix, sp.csr_matrix, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    加载LastFM数据集
    
    Args:
        data_path: 数据路径
        
    Returns:
        user_item_matrix: 用户-物品交互矩阵
        social_matrix: 社交关系矩阵
        user_features: 用户特征 (可选)
        item_features: 物品特征 (可选)
    """
    # 设置路径
    lastfm_dir = os.path.join(data_path, 'hetrec2011-lastfm-2k')
    user_artists_file = os.path.join(lastfm_dir, 'user_artists.dat')
    user_friends_file = os.path.join(lastfm_dir, 'user_friends.dat')
    
    # 加载用户-艺术家交互数据
    user_artists_df = pd.read_csv(user_artists_file, sep='\t', skiprows=0)
    
    # 加载用户社交关系数据
    user_friends_df = pd.read_csv(user_friends_file, sep='\t', skiprows=0)
    
    # 获取用户数和艺术家数
    unique_users = sorted(set(user_artists_df['userID'].unique()).union(
                         set(user_friends_df['userID'].unique())))
    unique_artists = sorted(user_artists_df['artistID'].unique())
    
    # 重新映射ID (确保用户和艺术家ID从0开始)
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
    
    # 使用权重作为交互强度
    rows = user_artists_mapped['userID'].values
    cols = user_artists_mapped['artistID'].values
    data = user_artists_mapped['weight'].values
    
    # 归一化权重
    max_weight = np.max(data)
    data = data / max_weight
    
    # 创建稀疏矩阵
    user_item_matrix = sp.csr_matrix((data, (rows, cols)), shape=(num_users, num_items))
    
    # 创建社交关系矩阵 (稀疏)
    rows = user_friends_mapped['userID'].values
    cols = user_friends_mapped['friendID'].values
    data = np.ones_like(rows)
    
    social_matrix = sp.csr_matrix((data, (rows, cols)), shape=(num_users, num_users))
    
    # 对称化社交矩阵(确保双向关系)
    social_matrix = social_matrix + social_matrix.T
    social_matrix.data = np.ones_like(social_matrix.data)  # 将值重置为1
    
    # 由于数据集中没有明确的用户和物品特征，返回None
    user_features = None
    item_features = None
    
    # 考虑使用社交网络作为用户特征
    # 使用用户的社交连接度作为特征
    if False:  # 设置为True以启用此功能
        user_deg = np.array(social_matrix.sum(axis=1)).flatten()
        user_features = np.zeros((num_users, 1))
        user_features[:, 0] = user_deg / np.max(user_deg)  # 归一化
    
    return user_item_matrix, social_matrix, user_features, item_features

def preprocess_adj_matrix(adj: sp.spmatrix) -> sp.spmatrix:
    """
    预处理邻接矩阵：进行度归一化
    
    Args:
        adj: 邻接矩阵 (可以是方阵或矩形矩阵)
        
    Returns:
        处理后的邻接矩阵
    """
    # 对于方阵（用户-用户或物品-物品），可以添加自环
    if adj.shape[0] == adj.shape[1]:
        adj = adj + sp.eye(adj.shape[0])
    
    # 行归一化 (针对行进行归一化)
    rowsum = np.array(adj.sum(axis=1)).flatten()
    # 避免除零
    rowsum[rowsum == 0] = 1.0
    r_inv = 1.0 / rowsum
    r_mat_inv = sp.diags(r_inv)
    # 行归一化结果
    adj_normalized = r_mat_inv.dot(adj)
    
    return adj_normalized 