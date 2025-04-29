import os
import time
import random
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_score, recall_score, ndcg_score
import torch.nn.functional as F

from model import LiteEIISRS


# 设置随机种子以保证实验可重复性
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    

# 加载LastFM数据集
def load_lastfm_data(data_dir):
    print("正在加载LastFM数据集...")
    
    # 读取用户-艺术家交互数据
    user_artists_file = os.path.join(data_dir, 'user_artists.dat')
    user_artists_df = pd.read_csv(user_artists_file, sep='\t', skiprows=1, 
                                names=['userID', 'artistID', 'weight'])
    
    # 读取用户好友数据
    user_friends_file = os.path.join(data_dir, 'user_friends.dat')
    user_friends_df = pd.read_csv(user_friends_file, sep='\t', skiprows=1,
                               names=['userID', 'friendID'])
    
    # 获取用户ID列表
    user_ids = sorted(user_artists_df['userID'].unique())
    artist_ids = sorted(user_artists_df['artistID'].unique())
    
    # 创建ID映射
    user_id_map = {old_id: new_id for new_id, old_id in enumerate(user_ids)}
    artist_id_map = {old_id: new_id for new_id, old_id in enumerate(artist_ids)}
    
    # 映射到连续ID
    user_artists_df['userID'] = user_artists_df['userID'].map(user_id_map)
    user_artists_df['artistID'] = user_artists_df['artistID'].map(artist_id_map)
    user_friends_df['userID'] = user_friends_df['userID'].map(user_id_map)
    user_friends_df['friendID'] = user_friends_df['friendID'].map(user_id_map)
    
    # 二值化交互数据 - 有交互为1，无交互为0
    user_artists_df['interaction'] = 1
    
    # 创建稀疏交互矩阵
    n_users = len(user_id_map)
    n_items = len(artist_id_map)
    
    interaction_matrix = sp.coo_matrix(
        (user_artists_df['interaction'], (user_artists_df['userID'], user_artists_df['artistID'])),
        shape=(n_users, n_items)
    ).tocsr()
    
    # 创建稀疏社交矩阵
    social_matrix = sp.coo_matrix(
        (np.ones(len(user_friends_df)), (user_friends_df['userID'], user_friends_df['friendID'])),
        shape=(n_users, n_users)
    ).tocsr()
    
    # 获取权重作为统计特征
    weight_matrix = sp.coo_matrix(
        (user_artists_df['weight'], (user_artists_df['userID'], user_artists_df['artistID'])),
        shape=(n_users, n_items)
    ).tocsr()
    
    print(f"数据集统计信息:")
    print(f"用户数量: {n_users}")
    print(f"艺术家数量: {n_items}")
    print(f"交互数量: {len(user_artists_df)}")
    print(f"好友关系数量: {len(user_friends_df)}")
    print(f"交互矩阵密度: {len(user_artists_df) / (n_users * n_items):.6f}")
    
    return interaction_matrix, social_matrix, weight_matrix, n_users, n_items


# 训练/测试集划分
def train_test_split(interaction_matrix, test_ratio=0.2, seed=42):
    np.random.seed(seed)
    
    # 获取所有非零元素坐标
    train_matrix = interaction_matrix.copy()
    test_matrix = sp.dok_matrix(interaction_matrix.shape)
    
    # 为每个用户抽取test_ratio比例的交互作为测试集
    for u in range(interaction_matrix.shape[0]):
        items = interaction_matrix[u].nonzero()[1]
        
        if len(items) >= 5:  # 确保用户至少有5个交互
            n_test_items = max(1, int(len(items) * test_ratio))
            test_items = np.random.choice(items, size=n_test_items, replace=False)
            
            # 从训练集移除测试集数据
            for i in test_items:
                train_matrix[u, i] = 0
                test_matrix[u, i] = 1
    
    return train_matrix.tocsr(), test_matrix.tocsr()


# 创建特征
def create_features(interaction_matrix, social_matrix, weight_matrix):
    """创建用户和物品的统计特征"""
    
    n_users, n_items = interaction_matrix.shape
    
    print("创建统计特征...")
    
    # 特征数量，决定了feature_dim
    n_features = 4
    
    # 用户特征: 交互次数、平均权重、最高权重、好友数
    user_features = np.zeros((n_users, n_features))
    
    # 交互次数
    user_features[:, 0] = np.array(interaction_matrix.sum(axis=1)).squeeze()
    
    # 用户平均和最高权重
    for u in range(n_users):
        weights = weight_matrix[u].data
        if len(weights) > 0:
            user_features[u, 1] = np.mean(weights)  # 平均权重
            user_features[u, 2] = np.max(weights)   # 最高权重
    
    # 好友数
    user_features[:, 3] = np.array(social_matrix.sum(axis=1)).squeeze()
    
    # 归一化
    user_features = normalize_features(user_features)
    
    # 物品特征: 被交互次数、平均权重、最高权重、流行度
    item_features = np.zeros((n_items, n_features))
    
    # 被交互次数
    item_features[:, 0] = np.array(interaction_matrix.sum(axis=0)).squeeze()
    
    # 物品平均和最高权重
    for i in range(n_items):
        weights = weight_matrix[:, i].data
        if len(weights) > 0:
            item_features[i, 1] = np.mean(weights)  # 平均权重
            item_features[i, 2] = np.max(weights)   # 最高权重
    
    # 流行度 (归一化的交互次数)
    item_features[:, 3] = item_features[:, 0] / (np.max(item_features[:, 0]) + 1e-8)
    
    # 归一化
    item_features = normalize_features(item_features)
    
    print(f"原始特征形状 - 用户特征: {user_features.shape}, 物品特征: {item_features.shape}")
    
    # 扩展特征
    user_features_ext = expand_features(user_features)
    item_features_ext = expand_features(item_features)
    
    print(f"扩展后特征形状 - 用户特征: {user_features_ext.shape}, 物品特征: {item_features_ext.shape}")
    print(f"特征统计: 最小值={np.min(user_features_ext)}, 最大值={np.max(user_features_ext)}, 均值={np.mean(user_features_ext)}")
    
    return user_features_ext, item_features_ext


# 特征归一化
def normalize_features(features):
    """Min-Max归一化特征"""
    min_vals = np.min(features, axis=0)
    max_vals = np.max(features, axis=0)
    
    # 处理常量特征
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    
    normalized_features = (features - min_vals) / range_vals
    
    # 将NaN值替换为0
    normalized_features = np.nan_to_num(normalized_features)
    
    return normalized_features


# 扩展特征
def expand_features(features):
    """
    扩展原始特征，创建多项式特征
    """
    n_samples, n_features = features.shape
    
    # 创建一个输出数组，包含：
    # 1. 原始特征
    # 2. 特征平方项
    # 3. 特征交叉项
    # 4. 偏置项
    
    # 计算输出特征数
    n_out_features = n_features + n_features + (n_features * (n_features - 1)) // 2 + 1
    
    # 创建扩展特征数组
    expanded = np.zeros((n_samples, n_out_features))
    
    # 填充原始特征
    expanded[:, :n_features] = features
    
    # 填充平方项
    expanded[:, n_features:2*n_features] = features ** 2
    
    # 填充特征交叉项
    idx = 2 * n_features
    for i in range(n_features):
        for j in range(i+1, n_features):
            expanded[:, idx] = features[:, i] * features[:, j]
            idx += 1
    
    # 填充偏置项
    expanded[:, -1] = 1.0
    
    return expanded


# 创建训练数据集
class LastFMTrainDataset(Dataset):
    def __init__(self, interaction_matrix, user_features, item_features, neg_ratio=4):
        self.interaction_matrix = interaction_matrix
        self.user_features = user_features
        self.item_features = item_features
        self.neg_ratio = neg_ratio
        
        # 获取所有正样本对
        self.user_item_pairs = []
        for u in range(interaction_matrix.shape[0]):
            items = interaction_matrix[u].nonzero()[1]
            for i in items:
                self.user_item_pairs.append((u, i))
                
        # 打印数据集一些信息
        print(f"训练数据集: 用户特征形状={user_features.shape}, 物品特征形状={item_features.shape}")
        print(f"正样本对数量: {len(self.user_item_pairs)}")
    
    def __len__(self):
        return len(self.user_item_pairs) * (1 + self.neg_ratio)
    
    def __getitem__(self, idx):
        # 确定是正样本还是负样本
        is_negative = idx % (1 + self.neg_ratio) != 0
        pair_idx = idx // (1 + self.neg_ratio)
        
        if not is_negative:  # 正样本
            u, i = self.user_item_pairs[pair_idx]
            label = 1.0
        else:  # 负样本
            u, i_pos = self.user_item_pairs[pair_idx]
            # 随机采样负样本
            while True:
                i = np.random.randint(0, self.interaction_matrix.shape[1])
                if self.interaction_matrix[u, i] == 0:  # 确保是负样本
                    break
            label = 0.0
        
        # 获取特征
        u_features = self.user_features[u]
        i_features = self.item_features[i]
        
        return u, i, u_features, i_features, label


# 调整评估批次大小，避免OOM
@torch.no_grad() # Ensure no gradients are computed during evaluation
def batch_evaluate(model, test_matrix, user_features, item_features, user_item_dict,
                  top_k=10, device='cpu', batch_size=32):
    """Optimized evaluation function: Ranks all items per user."""
    model.eval()

    n_users = test_matrix.shape[0]
    # Use item_features shape for n_items as test_matrix might not cover all items
    n_items = item_features.shape[0]

    # Convert features to tensors once, if possible and fits in memory
    # If features are too large, load them per batch
    try:
        user_features_tensor = torch.FloatTensor(user_features).to(device)
        item_features_tensor = torch.FloatTensor(item_features).to(device)
        all_item_ids_tensor = torch.arange(n_items, device=device)
        features_on_device = True
        print("User and item features loaded to GPU memory for evaluation.")
    except RuntimeError: # Likely CUDA OOM
        print("Warning: Features too large for GPU memory. Loading features per batch.")
        user_features_tensor = torch.FloatTensor(user_features) # Keep on CPU
        item_features_tensor = torch.FloatTensor(item_features) # Keep on CPU
        all_item_ids_tensor = torch.arange(n_items) # Keep on CPU
        features_on_device = False


    all_user_metrics = defaultdict(list) # Store metrics per user {metric_name: [user1_val, user2_val, ...]}
    evaluated_users_count = 0

    # Process users in batches
    pbar = tqdm(range(0, n_users, batch_size), desc="Evaluating User Batches")
    for user_start_idx in pbar:
        user_end_idx = min(user_start_idx + batch_size, n_users)
        batch_user_ids = list(range(user_start_idx, user_end_idx))
        current_batch_size = len(batch_user_ids)
        if current_batch_size == 0:
            continue

        # Prepare batch tensors
        batch_user_ids_tensor = torch.LongTensor(batch_user_ids).to(device)

        # Select user features for the batch
        if features_on_device:
            batch_u_features = user_features_tensor[batch_user_ids_tensor]
        else:
            # Load features for this batch and move to device
            batch_u_features = user_features_tensor[batch_user_ids].to(device)

        # Expand tensors for prediction against all items
        # Shape: (current_batch_size * n_items)
        batch_users_expanded = batch_user_ids_tensor.repeat_interleave(n_items)
        # Shape: (current_batch_size * n_items, feature_dim)
        batch_u_features_expanded = batch_u_features.repeat_interleave(n_items, dim=0)
        # Shape: (current_batch_size * n_items)
        batch_items_expanded = all_item_ids_tensor.repeat(current_batch_size)
         # Shape: (current_batch_size * n_items, feature_dim)
        if features_on_device:
            batch_i_features_expanded = item_features_tensor.repeat(current_batch_size, 1)
        else:
            # If item features are on CPU, repeat and move to device
             batch_i_features_expanded = item_features_tensor.to(device).repeat(current_batch_size, 1)


        # Predict scores for the batch of users against all items
        # This might still be memory intensive if batch_size * n_items is large
        try:
            # Assuming model.predict handles features correctly and returns raw scores
            all_batch_scores = model.predict(batch_users_expanded,
                                             batch_items_expanded,
                                             batch_u_features_expanded,
                                             batch_i_features_expanded)

            # Reshape scores: (current_batch_size, n_items)
            all_batch_scores = all_batch_scores.view(current_batch_size, n_items)
            all_batch_scores_cpu = all_batch_scores.cpu().numpy()

        except RuntimeError as e:
             if "CUDA out of memory" in str(e):
                 print(f"\nOOM Error during batch prediction (batch_size={current_batch_size}, n_items={n_items}). Try reducing eval_batch_size.")
                 # Option: Fallback to per-user prediction within the batch? Or just raise error.
                 # For now, let's just report and continue to see if other batches work
                 continue # Skip this batch
             else:
                 # Print detailed error and traceback for other RuntimeErrors
                 print(f"\nRuntime error during prediction for batch {user_start_idx}-{user_end_idx}: {e}")
                 import traceback
                 traceback.print_exc()
                 continue # Skip this batch
        except Exception as e:
            # Catch other potential errors during prediction
            print(f"\nUnexpected error during prediction for batch {user_start_idx}-{user_end_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue # Skip this batch

        # Process scores for each user in the batch
        batch_evaluated_count = 0
        for i, u in enumerate(batch_user_ids):
            user_scores = all_batch_scores_cpu[i] # Scores for user u against all items

            # Get user's test and train items
            test_pos_items = test_matrix[u].nonzero()[1]
            if len(test_pos_items) == 0:
                continue # Skip user if no items in test set

            train_items = user_item_dict.get(u, np.array([]))

            # Set scores of training items to -inf
            # Use valid indices for train_items
            valid_train_items = train_items[train_items < n_items] # Ensure indices are within bounds
            if len(valid_train_items) > 0:
                 try:
                    user_scores[valid_train_items] = -np.inf
                 except IndexError:
                     print(f"Warning: IndexError when masking train items for user {u}. Max train item index: {np.max(valid_train_items) if len(valid_train_items)>0 else 'N/A'}, n_items: {n_items}")
                     # Handle potential index out of bounds if item IDs are inconsistent
                     pass # Continue evaluation for this user, but train items might not be masked

            # Get top-k recommendations indices
            # Use np.argpartition for efficiency if only top-k scores/indices are needed
            # For simplicity, we use argsort here which sorts everything.
            # If k is small compared to n_items, argpartition is faster.
            # top_k_indices = np.argpartition(user_scores, -top_k)[-top_k:]
            # top_k_indices = top_k_indices[np.argsort(user_scores[top_k_indices])][::-1]
            top_k_indices = np.argsort(user_scores)[::-1][:top_k]


            # Calculate metrics for this user
            # Ensure test_pos_items are also valid indices
            valid_test_pos_items = test_pos_items[test_pos_items < n_items]
            if len(valid_test_pos_items) == 0:
                 # This case should ideally not happen if test matrix comes from valid items
                 continue

            hits_mask = np.isin(top_k_indices, valid_test_pos_items) # Boolean array: True if test item is in top-k
            num_hits = np.sum(hits_mask)
            batch_evaluated_count += 1 # Count users actually evaluated in this batch

            # --- Metric Calculation ---
            # Hit Rate (User-level: 1 if any hit, 0 otherwise)
            user_hit_rate = 1.0 if num_hits > 0 else 0.0
            all_user_metrics[f'HR@{top_k}'].append(user_hit_rate)

            if num_hits > 0:
                # Precision@k = (# hits in top-k) / k
                precision = num_hits / top_k
                all_user_metrics[f'Precision@{top_k}'].append(precision)

                # Recall@k = (# hits in top-k) / (# test items)
                recall = num_hits / len(valid_test_pos_items)
                all_user_metrics[f'Recall@{top_k}'].append(recall)

                # NDCG@k
                # Create relevance array (1 if item in top-k is relevant, 0 otherwise)
                relevance = hits_mask.astype(float)
                # DCG = sum(relevance[i] / log2(i + 2)) for i in 0..k-1
                dcg = np.sum(relevance / np.log2(np.arange(2, top_k + 2)))
                # IDCG = sum(1 / log2(i + 2)) for i in 0..min(k, num_relevant)-1
                num_relevant = len(valid_test_pos_items)
                idcg = np.sum(1. / np.log2(np.arange(2, min(top_k, num_relevant) + 2)))
                ndcg = dcg / idcg if idcg > 0 else 0
                all_user_metrics[f'NDCG@{top_k}'].append(ndcg)

            else: # No hits for this user
                all_user_metrics[f'Precision@{top_k}'].append(0.0)
                all_user_metrics[f'Recall@{top_k}'].append(0.0)
                all_user_metrics[f'NDCG@{top_k}'].append(0.0)
        # Update total evaluated user count
        evaluated_users_count += batch_evaluated_count
        pbar.set_postfix({"Evaluated": f"{evaluated_users_count}/{n_users}"})


    # Calculate average metrics across all users evaluated
    final_metrics = {}
    if evaluated_users_count > 0:
         print(f"\nEvaluation Summary: Evaluated {evaluated_users_count} users with test data.")
         for metric, values in all_user_metrics.items():
             if values: # Ensure list is not empty before calculating mean
                 final_metrics[metric] = np.mean(values)
             else:
                 final_metrics[metric] = 0.0 # Default to 0 if no values recorded
    else:
         print("\nWarning: No users with test data were successfully evaluated.")
         # Return zeros or empty dict? Let's return zeros for consistency.
         for metric_name in [f'HR@{top_k}', f'NDCG@{top_k}', f'Precision@{top_k}', f'Recall@{top_k}']:
             final_metrics[metric_name] = 0.0

    # Clear CUDA cache if features were loaded to GPU batch-wise to potentially free memory
    if not features_on_device and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning: Error clearing CUDA cache: {e}")

    return final_metrics


# 训练函数
def train(model, train_loader, optimizer, epoch, device, log_interval=100):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    n_batches = len(train_loader)
    
    # 使用tqdm显示进度条
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    first_batch = True
    
    for batch_idx, (u, i, u_f, i_f, labels) in enumerate(pbar):
        # 将数据移至设备
        u, i = u.to(device), i.to(device)
        u_f, i_f = u_f.to(device), i_f.to(device)
        labels = labels.to(device)
        
        try:
            if first_batch:
                # 在第一个批次打印形状信息，帮助调试
                print(f"第一个批次: u.shape={u.shape}, i.shape={i.shape}")
                print(f"用户特征: u_f.shape={u_f.shape}, 类型={u_f.dtype}")
                print(f"物品特征: i_f.shape={i_f.shape}, 类型={i_f.dtype}")
                first_batch = False
                
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            u_emb, i_emb = model(u, i, u_f, i_f)
            
            # 计算预测评分和损失
            predictions = torch.sum(u_emb * i_emb, dim=1)
            # 使用BCE损失
            bce_loss = F.binary_cross_entropy_with_logits(predictions, labels)
            # 加上正则化损失
            reg_loss = model.reg_loss()
            loss = bce_loss + reg_loss
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
            
            # 更新进度条
            pbar.set_postfix(loss=f"{loss.item():.4f}", bce=f"{bce_loss.item():.4f}", reg=f"{reg_loss.item():.4f}")
        
        except Exception as e:
            print(f"批次 {batch_idx} 处理失败: {e}")
            # 提供更详细的错误信息
            if u_f is not None and i_f is not None:
                print(f"u_f.shape={u_f.shape}, i_f.shape={i_f.shape}")
                
            import traceback
            traceback.print_exc()
            continue  # 继续处理下一个批次
    
    # 返回平均损失
    avg_loss = total_loss / n_batches
    print(f"Epoch {epoch} 平均损失: {avg_loss:.4f}")
    
    return avg_loss


# 可视化训练过程
def plot_metrics(train_metrics, val_metrics, k, save_path='metrics.png'):
    plt.figure(figsize=(20, 5))

    # 绘制训练损失
    plt.subplot(1, 4, 1)
    plt.plot(train_metrics['loss'], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)

    # 绘制 HR@k
    plt.subplot(1, 4, 2)
    metric_key = f'HR@{k}'
    if metric_key in val_metrics and len(val_metrics[metric_key]) > 0:
        plt.plot(val_metrics[metric_key], label=metric_key)
    plt.xlabel('Epoch')
    plt.ylabel('HR')
    plt.title(f'Hit Ratio @{k}')
    plt.legend()
    plt.grid(True)

    # 绘制 NDCG@k
    plt.subplot(1, 4, 3)
    metric_key = f'NDCG@{k}'
    if metric_key in val_metrics and len(val_metrics[metric_key]) > 0:
        plt.plot(val_metrics[metric_key], label=metric_key)
    plt.xlabel('Epoch')
    plt.ylabel('NDCG')
    plt.title(f'NDCG @{k}')
    plt.legend()
    plt.grid(True)

    # 绘制 Recall@k (或 Precision@k - 选择一个或添加子图)
    plt.subplot(1, 4, 4)
    metric_key_recall = f'Recall@{k}'
    metric_key_precision = f'Precision@{k}'
    plotted = False
    if metric_key_recall in val_metrics and len(val_metrics[metric_key_recall]) > 0:
        plt.plot(val_metrics[metric_key_recall], label=metric_key_recall)
        plt.ylabel('Recall')
        plt.title(f'Recall @{k}')
        plotted = True
    elif metric_key_precision in val_metrics and len(val_metrics[metric_key_precision]) > 0:
        plt.plot(val_metrics[metric_key_precision], label=metric_key_precision)
        plt.ylabel('Precision')
        plt.title(f'Precision @{k}')
        plotted = True

    if plotted:
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
    else:
        plt.title(f'Recall/Precision @{k} (N/A)')
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# 保存实验结果
def save_results(args, metrics, elapsed_time, save_path='results.txt'):
    with open(save_path, 'w') as f:
        f.write(f"实验参数:\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        
        f.write(f"\n最终评价指标:\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
        
        f.write(f"\n训练时间: {elapsed_time:.2f}秒\n")


# 主函数
def main():
    parser = argparse.ArgumentParser(description='Lite-EIISRS模型训练')
    parser.add_argument('--data_dir', type=str, default='data/hetrec2011-lastfm-2k', help='数据集路径')
    parser.add_argument('--embed_dim', type=int, default=64, help='基础嵌入和最终输出维度')
    parser.add_argument('--batch_size', type=int, default=1024, help='批次大小')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='评估时的用户批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮次')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--n_layers_light', type=int, default=2, help='LightGCN层数')
    parser.add_argument('--n_layers_social', type=int, default=2, help='SocialGCN层数')
    parser.add_argument('--n_heads', type=int, default=2, help='注意力头数')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率 (模型内部未使用)')
    parser.add_argument('--beta', type=float, default=0.2, help='最终残差连接系数')
    parser.add_argument('--eval_interval', type=int, default=1, help='评估间隔')
    parser.add_argument('--top_k', type=int, default=10, help='推荐top-k物品')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载器的工作进程数')
    parser.add_argument('--use_gpu', action='store_true', help='是否使用GPU')
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    interaction_matrix, social_matrix, weight_matrix, n_users, n_items = load_lastfm_data(args.data_dir)
    
    # 训练集/测试集划分
    train_matrix, test_matrix = train_test_split(interaction_matrix, test_ratio=0.2, seed=args.seed)
    
    # 创建训练集中的用户-物品字典
    user_item_dict = {}
    for u in range(n_users):
        items = train_matrix[u].nonzero()[1]
        user_item_dict[u] = items
    
    # 创建特征 (得到扩展后的特征)
    user_features, item_features = create_features(train_matrix, social_matrix, weight_matrix)
    feature_dim = user_features.shape[1] # 获取扩展后的特征维度
    print(f"扩展后特征维度: {feature_dim}")
    
    # 创建数据集和数据加载器
    train_dataset = LastFMTrainDataset(train_matrix, user_features, item_features, neg_ratio=4)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                            num_workers=args.num_workers, pin_memory=args.use_gpu)
    
    try:
        # 初始化模型 (传递所有需要的参数)
        model = LiteEIISRS(
            n_users=n_users,
            n_items=n_items,
            embed_dim=args.embed_dim,
            interaction_matrix=train_matrix,
            social_matrix=social_matrix,
            feature_dim=feature_dim,  # 传递扩展后的维度
            n_layers_light=args.n_layers_light,
            n_layers_social=args.n_layers_social,
            n_heads=args.n_heads,
            beta=args.beta # 传递 beta
        ).to(device)
        
        # 优化器
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        
        # 记录训练指标
        train_metrics = {'loss': []}
        val_metrics = defaultdict(list)
        
        # 训练开始时间
        start_time = time.time()
        
        # 训练循环
        for epoch in range(1, args.epochs + 1):
            # 训练
            train_loss = train(model, train_loader, optimizer, epoch, device)
            if train_loss is None: # 如果训练失败，提前退出
                 print(f"Epoch {epoch} 训练失败，提前终止。")
                 break
            train_metrics['loss'].append(train_loss)

            # 更新学习率
            scheduler.step(train_loss)

            # 评估
            if epoch % args.eval_interval == 0:
                print(f"\nEpoch {epoch}: Evaluating with Top-K={args.top_k}...")
                try:
                    # 使用优化的 batch_evaluate 函数
                    # 它计算由 args 指定的单个 top_k 的指标
                    metrics = batch_evaluate(model, test_matrix, user_features, item_features,
                                             user_item_dict, top_k=args.top_k, device=device,
                                             batch_size=args.eval_batch_size)

                    # 记录评估指标
                    if metrics:
                        print(f"Epoch {epoch} Evaluation Results (Top-{args.top_k}):")
                        for key, value in metrics.items():
                            val_metrics[key].append(value) # 为此 epoch 追加指标值
                            print(f"{key}: {value:.4f}")
                    else:
                         print(f"Epoch {epoch} Evaluation failed or returned no metrics.")
                         # 如果需要绘图，追加零或进行适当处理
                         for metric_name in [f'HR@{args.top_k}', f'NDCG@{args.top_k}', f'Precision@{args.top_k}', f'Recall@{args.top_k}']:
                              val_metrics[metric_name].append(0.0)


                except Exception as e:
                    print(f"Evaluation failed for epoch {epoch}: {e}")
                    import traceback
                    traceback.print_exc()
                    # 追加零或进行适当处理
                    for metric_name in [f'HR@{args.top_k}', f'NDCG@{args.top_k}', f'Precision@{args.top_k}', f'Recall@{args.top_k}']:
                         val_metrics[metric_name].append(0.0)

            # 每 5 个 epoch 或最后一个 epoch 绘制一次指标图
            if epoch % 5 == 0 or epoch == args.epochs:
                try:
                    print(f"Plotting metrics for epoch {epoch}...")
                    plot_metrics(train_metrics, val_metrics, k=args.top_k, # 传递 k 值
                             save_path=os.path.join(args.output_dir, f'metrics_epoch_{epoch}.png'))
                except Exception as e:
                    print(f"Plotting metrics failed: {e}")

        # 训练结束时间
        end_time = time.time()
        elapsed_time = end_time - start_time

        # 获取最终指标
        final_metrics = {}
        for key, values in val_metrics.items():
             if values: # 检查列表是否不为空
                 final_metrics[key] = values[-1] # 获取最后记录的值

        # 保存最终指标和模型
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'model.pt'))
        save_results(args, final_metrics, elapsed_time,
                     save_path=os.path.join(args.output_dir, 'results.txt'))


        print(f"\nTraining finished! Elapsed time: {elapsed_time:.2f} seconds")
        if final_metrics:
            print("\nFinal Evaluation Metrics:")
            # 打印所有最终指标
            sorted_metrics = sorted(final_metrics.items())
            for key, value in sorted_metrics:
                 # 检查 value 是否为数字类型，以应用格式化
                 if isinstance(value, (int, float)):
                      print(f"{key}: {value:.4f}")
                 else:
                      print(f"{key}: {value}") # 原样打印非数字值
            # # 示例特定打印 (可选):
            # hr_final = final_metrics.get(f'HR@{args.top_k}', 'N/A')
            # ndcg_final = final_metrics.get(f'NDCG@{args.top_k}', 'N/A')
            # if isinstance(hr_final, float) and isinstance(ndcg_final, float):
            #      print(f"Final Top-{args.top_k}: HR={hr_final:.4f}, NDCG={ndcg_final:.4f}")
            # else:
            #      print(f"Final Top-{args.top_k} metrics not available.")
        else:
            print("\nWarning: No final metrics were recorded.")

    except Exception as e:
        print(f"\nInitialization or training process encountered a critical error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 