import torch
import numpy as np
import scipy.sparse as sp
import os
from config import Config
from logger import Logger
from train import Trainer
from social_recommender import SocialRecommender
from utils.data_utils import create_data_loaders
from utils.data_loader import load_lastfm_data, preprocess_adj_matrix

def sparse_scipy_to_torch(scipy_sparse_matrix):
    """将scipy稀疏矩阵转换为PyTorch稀疏张量"""
    # 获取坐标和值
    coords = scipy_sparse_matrix.nonzero()
    values = scipy_sparse_matrix.data
    
    # 创建索引张量 [2, nnz]
    indices = torch.LongTensor([coords[0], coords[1]])
    
    # 创建PyTorch稀疏张量
    torch_sparse = torch.sparse.FloatTensor(
        indices, 
        torch.FloatTensor(values),
        torch.Size(scipy_sparse_matrix.shape)
    )
    
    return torch_sparse

def main():
    # 创建配置
    config = Config()
    
    # 修改配置参数
    config.model.embed_dim = 64
    config.model.n_layers = 2
    config.model.dropout = 0.1
    config.model.temperature = 0.2
    
    config.training.batch_size = 256
    config.training.learning_rate = 0.001
    config.training.weight_decay = 1e-5
    config.training.num_epochs = 100
    config.training.early_stopping_patience = 10
    config.training.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config.data.num_negatives = 4
    
    # 创建日志记录器
    logger = Logger(config.training.log_dir)
    
    # 记录配置
    logger.log_hyperparameters({
        'model': vars(config.model),
        'training': vars(config.training),
        'data': vars(config.data)
    })
    
    # 加载数据
    logger.log_text("加载数据...")
    user_item_matrix, social_matrix, user_features, item_features = load_lastfm_data(config.data.data_path)
    
    # 预处理邻接矩阵
    logger.log_text("预处理邻接矩阵...")
    ui_adj = preprocess_adj_matrix(user_item_matrix)
    iu_adj = preprocess_adj_matrix(user_item_matrix.T)
    social_adj = preprocess_adj_matrix(social_matrix) if social_matrix is not None else None
    
    # 创建数据加载器
    logger.log_text("创建数据加载器...")
    train_loader, val_loader, test_loader = create_data_loaders(
        user_item_matrix=user_item_matrix,
        social_matrix=social_matrix,
        user_features=user_features,
        item_features=item_features,
        batch_size=config.training.batch_size,
        num_negatives=config.data.num_negatives,
        num_workers=config.training.num_workers,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        random_seed=config.data.random_seed
    )
    
    # 创建模型
    logger.log_text("初始化模型...")
    model = SocialRecommender(
        num_users=user_item_matrix.shape[0],
        num_items=user_item_matrix.shape[1],
        embed_dim=config.model.embed_dim,
        interaction_matrix=user_item_matrix,
        user_features=user_features,
        item_features=item_features,
        n_layers=config.model.n_layers,
        dropout=config.model.dropout,
        temperature=config.model.temperature
    )
    
    # 输出模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    logger.log_text(f"模型参数总数: {total_params:,}")
    
    # 创建训练器
    trainer = Trainer(config, logger)
    
    # 将稀疏矩阵转换为PyTorch稀疏张量
    logger.log_text("准备邻接矩阵...")
    ui_adj_torch = sparse_scipy_to_torch(ui_adj)
    iu_adj_torch = sparse_scipy_to_torch(iu_adj)
    social_adj_torch = sparse_scipy_to_torch(social_adj) if social_adj is not None else None
    
    # 开始训练
    logger.log_text("开始训练...")
    trainer.train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        ui_adj=ui_adj_torch,
        iu_adj=iu_adj_torch,
        social_adj=social_adj_torch
    )
    
    logger.log_text("训练完成!")

if __name__ == "__main__":
    main() 