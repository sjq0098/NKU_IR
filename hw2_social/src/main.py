import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.dataset import LastFMDataset
from models.eiisrs import EIISRS
from utils.trainer import Trainer
from config.config import Config
import os
import time

def main():
    # 设置随机种子
    torch.manual_seed(42)
    
    # 加载配置
    config = Config()
    
    # 创建数据目录
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    # 加载数据集
    train_dataset = LastFMDataset(config, is_train=True)
    val_dataset = LastFMDataset(config, is_train=False)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=train_dataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=val_dataset.collate_fn
    )
    
    # 创建模型
    model = EIISRS(
        num_users=train_dataset.num_users,
        num_items=train_dataset.num_items,
        embedding_dim=config.EMBEDDING_DIM,
        social_conv_layers=config.SOCIAL_CONV_LAYERS
    )
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # 创建训练器
    trainer = Trainer(model, train_loader, val_loader, optimizer, config)
    
    # 记录开始时间
    start_time = time.time()
    
    # 训练模型
    print("开始训练...")
    best_metric = trainer.train(config.NUM_EPOCHS)
    
    # 计算总训练时间
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print(f"\n训练完成!")
    print(f"总训练时间: {hours}小时 {minutes}分钟 {seconds}秒")
    print(f"最佳HR@10: {best_metric:.4f}")

if __name__ == "__main__":
    main()