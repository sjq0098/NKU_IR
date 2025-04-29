import numpy as np
import sys
import os
from typing import Tuple, List

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def calculate_metrics(ranked_items: np.ndarray, test_items: np.ndarray, k: int) -> Tuple[float, float]:
    """
    计算推荐系统评价指标
    
    参数:
        ranked_items: 排序后的预测物品列表
        test_items: 真实的测试物品列表
        k: 推荐列表的长度
        
    返回:
        hr: 命中率 (Hit Ratio)
        ndcg: 归一化折扣累积增益 (Normalized Discounted Cumulative Gain)
    """
    # 截断预测列表到k
    ranked_items = ranked_items[:k]
    
    # 计算命中率 (HR)
    hit = len(np.intersect1d(ranked_items, test_items))
    hr = 1.0 if hit > 0 else 0.0
    
    # 计算NDCG
    ndcg = 0.0
    if hit > 0:
        # 查找第一个命中的位置
        for i, item in enumerate(ranked_items):
            if item in test_items:
                ndcg = np.reciprocal(np.log2(i + 2))  # log2(i+2) 是因为位置从0开始
                break
    
    return hr, ndcg

def calculate_precision_recall(ranked_items: np.ndarray, test_items: np.ndarray, k: int) -> Tuple[float, float]:
    """
    计算精确率和召回率
    
    参数:
        ranked_items: 排序后的预测物品列表
        test_items: 真实的测试物品列表
        k: 推荐列表的长度
        
    返回:
        precision: 精确率
        recall: 召回率
    """
    # 截断预测列表到k
    ranked_items = ranked_items[:k]
    
    # 计算命中数
    hit = len(np.intersect1d(ranked_items, test_items))
    
    # 计算精确率
    precision = hit / min(k, len(ranked_items)) if len(ranked_items) > 0 else 0.0
    
    # 计算召回率
    recall = hit / len(test_items) if len(test_items) > 0 else 0.0
    
    return precision, recall

def calculate_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    计算AUC (Area Under the ROC Curve)
    
    参数:
        scores: 预测分数
        labels: 真实标签 (0/1)
        
    返回:
        auc: AUC值
    """
    # 正负样本数量
    n_pos = np.sum(labels)
    n_neg = len(labels) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return 0.5  # 当所有样本都是正样本或负样本时，AUC = 0.5
    
    # 获取排序索引
    indices = np.argsort(-scores)  # 降序排序
    
    # 排序后的标签
    sorted_labels = labels[indices]
    
    # 计算正样本的排名
    pos_ranks = np.where(sorted_labels == 1)[0]
    
    # 计算AUC
    auc = (np.sum(pos_ranks) - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)
    
    return auc 