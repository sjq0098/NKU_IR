import os
import logging
import random
import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def load_data(data_path, social_path=None):
    """加载数据
    
    Args:
        data_path: 用户-物品交互数据路径
        social_path: 社交关系数据路径
        
    Returns:
        R: 用户-物品交互稀疏矩阵 (CSR格式)
        S: 社交关系稀疏矩阵 (CSR格式)
        user_items_dict: 用户-物品字典
        num_users: 用户数量
        num_items: 物品数量
    """
    logger.info(f"加载交互数据: {data_path}")
    user_item_data = []
    with open(data_path, 'r') as f:
        # 跳过标题行
        first_line = f.readline().strip()
        is_header = False
        parts = first_line.split()
        if len(parts) >= 2:
            try:
                int(parts[0])
                int(parts[1])
                f.seek(0)
            except ValueError:
                is_header = True
                logger.info(f"检测到交互数据标题行: {first_line}")
        
        for line in f:
            items = line.strip().split()
            if len(items) >= 2:
                try:
                    user, item, *optional = items
                    user, item = int(user), int(item)
                    rating = float(optional[0]) if optional else 1.0
                    user_item_data.append((user, item, rating))
                except ValueError as e:
                    logger.warning(f"跳过无法解析的交互行: {line.strip()}, 错误: {e}")
    
    if not user_item_data:
        raise ValueError(f"未从 {data_path} 加载到有效数据。")
    
    num_users = max(data[0] for data in user_item_data) + 1
    num_items = max(data[1] for data in user_item_data) + 1
    logger.info(f"用户数量: {num_users}, 物品数量: {num_items}")
    
    user_indices = [data[0] for data in user_item_data]
    item_indices = [data[1] for data in user_item_data]
    ratings = [data[2] for data in user_item_data]
    R = sp.coo_matrix((ratings, (user_indices, item_indices)), shape=(num_users, num_items)).tocsr()
    
    user_items_dict = {}
    for user, item, _ in user_item_data:
        user_items_dict.setdefault(user, []).append(item)
    
    S = None
    if social_path and os.path.exists(social_path):
        logger.info(f"加载社交数据: {social_path}")
        social_data = []
        with open(social_path, 'r') as f:
            # 跳过标题行
            first_line = f.readline().strip()
            is_header = False
            parts = first_line.split()
            if len(parts) >= 2:
                try:
                    int(parts[0])
                    int(parts[1])
                    f.seek(0)
                except ValueError:
                    is_header = True
                    logger.info(f"检测到社交数据标题行: {first_line}")
            
            for line in f:
                items = line.strip().split()
                if len(items) >= 2:
                    try:
                        user1, user2 = int(items[0]), int(items[1])
                        if user1 < num_users and user2 < num_users:
                            social_data.append((user1, user2))
                            social_data.append((user2, user1))  # 双向关系
                    except ValueError:
                        logger.warning(f"跳过无法解析的社交行: {line.strip()}")
        
        if social_data:
            user1_indices = [data[0] for data in social_data]
            user2_indices = [data[1] for data in social_data]
            social_weights = [1.0] * len(social_data)
            S = sp.coo_matrix((social_weights, (user1_indices, user2_indices)), shape=(num_users, num_users)).tocsr()
            logger.info(f"社交关系数量: {len(social_data)//2}")
        else:
             logger.warning(f"未从 {social_path} 加载到有效社交数据。")
    else:
        logger.warning(f"未找到社交数据文件: {social_path} 或未提供社交数据路径。")

    return R, S, user_items_dict, num_users, num_items

def split_data(user_items_dict, R_shape, test_ratio=0.2, val_ratio=0.1, seed=42):
    """按用户划分训练集、验证集和测试集"""
    np.random.seed(seed)
    
    train_indices, val_indices, test_indices = [], [], []
    train_values, val_values, test_values = [], [], []
    train_user_items, val_user_items, test_user_items = {}, {}, {}
    
    for user, items in user_items_dict.items():
        n_items = len(items)
        if n_items < 3:  # 至少需要3个交互才能划分
            for item in items:
                train_indices.append((user, item))
                train_values.append(1.0) # 假设交互值为1
                train_user_items.setdefault(user, []).append(item)
            continue
        
        shuffled_items = np.random.permutation(items)
        
        n_test = max(1, int(n_items * test_ratio))
        n_val = max(1, int(n_items * val_ratio))
        n_train = n_items - n_test - n_val
        
        if n_train < 1: # 确保训练集至少有一个交互
             n_val = max(0, n_val - (1 - n_train)) # 减少验证集以补充训练集
             n_train = 1

        test_items_set = set(shuffled_items[:n_test])
        val_items_set = set(shuffled_items[n_test : n_test + n_val])
        
        for item in shuffled_items:
            if item in test_items_set:
                test_indices.append((user, item))
                test_values.append(1.0)
                test_user_items.setdefault(user, []).append(item)
            elif item in val_items_set:
                val_indices.append((user, item))
                val_values.append(1.0)
                val_user_items.setdefault(user, []).append(item)
            else:
                train_indices.append((user, item))
                train_values.append(1.0)
                train_user_items.setdefault(user, []).append(item)

    def create_csr_matrix(indices, values, shape):
        if not indices:
            return sp.csr_matrix(shape)
        rows, cols = zip(*indices)
        return sp.coo_matrix((values, (rows, cols)), shape=shape).tocsr()

    R_train_csr = create_csr_matrix(train_indices, train_values, R_shape)
    
    logger.info(f"数据集划分: 训练集交互数={len(train_indices)}, 验证集交互数={len(val_indices)}, 测试集交互数={len(test_indices)}")
    logger.info(f"训练集用户数={len(train_user_items)}, 验证集用户数={len(val_user_items)}, 测试集用户数={len(test_user_items)}")
    
    return R_train_csr, train_user_items, val_user_items, test_user_items

def create_normalized_adj_matrix(R):
    """创建归一化的邻接矩阵 (LightGCN使用)
       包括用户-物品，物品-用户，以及组合的双向图矩阵
    """
    num_users, num_items = R.shape
    logger.info("创建归一化邻接矩阵...")
    
    # 用户->物品 (R_norm)
    rowsum_user = np.array(R.sum(1)).flatten()
    d_inv_user = np.power(rowsum_user, -0.5)
    d_inv_user[np.isinf(d_inv_user)] = 0.
    d_mat_inv_user = sp.diags(d_inv_user)
    
    # 物品->用户 (R_item_norm, 注意这里用 R.T)
    R_T = R.T.tocsr()
    rowsum_item = np.array(R_T.sum(1)).flatten()
    d_inv_item = np.power(rowsum_item, -0.5)
    d_inv_item[np.isinf(d_inv_item)] = 0.
    d_mat_inv_item = sp.diags(d_inv_item)

    R_norm = d_mat_inv_user @ R @ d_mat_inv_item # 用户->物品
    R_item_norm = d_mat_inv_item @ R_T @ d_mat_inv_user # 物品->用户

    # 双向图矩阵 (用于某些GCN变体)
    adj_mat = sp.dok_matrix((num_users + num_items, num_users + num_items), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R_dok = R.todok()
    
    adj_mat[:num_users, num_users:] = R_dok
    adj_mat[num_users:, :num_users] = R_dok.T
    adj_mat = adj_mat.todok()
    
    rowsum = np.array(adj_mat.sum(axis=1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    
    norm_adj = d_mat.dot(adj_mat).dot(d_mat).tocsr()
    
    logger.info("归一化邻接矩阵创建完成。")
    return R_norm, R_item_norm, norm_adj

def create_normalized_social_matrix(S):
    """创建归一化的社交邻接矩阵"""
    if S is None:
        return None
    logger.info("创建归一化社交矩阵...")
    rowsum = np.array(S.sum(1)).flatten()
    d_inv = np.power(rowsum, -0.5)
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    norm_S = d_mat.dot(S).dot(d_mat).tocsr()
    logger.info("归一化社交矩阵创建完成。")
    return norm_S

def convert_sparse_matrix_to_sparse_tensor(X):
    """将scipy稀疏矩阵转换为torch稀疏张量"""
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    # 注意: torch.sparse.FloatTensor 在较新版本中可能被弃用，推荐使用 torch.sparse_coo_tensor
    # return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
    return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))

class SVDFeatureExtractor:
    """SVD特征提取器"""
    def __init__(self, k_components=64):
        self.k_components = k_components
        self.svd = TruncatedSVD(n_components=k_components, random_state=42)
        
    def fit_transform(self, R_train_csr):
        logger.info(f"提取SVD特征 (k={self.k_components})...")
        user_factors = self.svd.fit_transform(R_train_csr)
        item_factors = self.svd.components_.T
        explained_variance = self.svd.explained_variance_ratio_.sum()
        logger.info(f"SVD解释方差: {explained_variance:.4f}")
        return user_factors, item_factors

def evaluate(model, user_items_dict, user_embeds, item_embeds, topk_list=[5, 10, 20], device='cuda'):
    """评估模型性能"""
    model.eval()
    max_k = max(topk_list)
    
    precisions = {k: [] for k in topk_list}
    recalls = {k: [] for k in topk_list}
    ndcgs = {k: [] for k in topk_list}
    hit_ratios = {k: [] for k in topk_list}
    
    with torch.no_grad():
        for user, ground_truth_items in tqdm(user_items_dict.items(), desc="评估模型", leave=False, ncols=80):
            if not ground_truth_items: continue
            
            user_tensor = torch.LongTensor([user]).to(device)
            # 预测用户对所有物品的评分
            scores = model.predict(user_embeds, item_embeds, user_tensor).squeeze()
            
            # 获取Top-K推荐 (排除训练集物品 - 这里简化为不排除)
            _, top_indices = torch.topk(scores, k=max_k)
            recommended_items = top_indices.cpu().numpy()
            
            gt_set = set(ground_truth_items)
            
            for k in topk_list:
                rec_items_k = recommended_items[:k]
                hit_num = len(set(rec_items_k) & gt_set)
                
                precisions[k].append(hit_num / k)
                recalls[k].append(hit_num / len(gt_set))
                hit_ratios[k].append(1.0 if hit_num > 0 else 0.0)
                
                dcg = 0.0
                for i, item_idx in enumerate(rec_items_k):
                    if item_idx in gt_set:
                        dcg += 1.0 / np.log2(i + 2)
                idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(gt_set), k))])
                ndcgs[k].append(dcg / idcg if idcg > 0 else 0.0)
    
    metrics = {}
    for k in topk_list:
        metrics[f'Precision@{k}'] = np.mean(precisions[k])
        metrics[f'Recall@{k}'] = np.mean(recalls[k])
        metrics[f'NDCG@{k}'] = np.mean(ndcgs[k])
        metrics[f'HR@{k}'] = np.mean(hit_ratios[k])
    
    return metrics 