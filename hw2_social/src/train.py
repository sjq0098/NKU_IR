import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import argparse
import logging
import sys
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Tuple, Any, Optional
from tqdm import tqdm

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.dataset import LastFMDataset
from src.models.social_recommender import SocialRecommender
from src.utils.metrics import calculate_metrics

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SocialRec")


def train_model(args: argparse.Namespace):
    """训练模型"""
    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 设置输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    log_dir = os.path.join(args.output_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 设置TensorBoard
    writer = SummaryWriter(log_dir)
    
    # 加载数据集
    logger.info(f"加载数据集: {args.data_dir}")
    dataset = LastFMDataset(args.data_dir, device)
    
    # 分割训练集和测试集
    train_matrix, test_matrix = dataset.get_train_test_split(args.test_ratio)
    logger.info(f"训练集: {train_matrix.nnz}条交互, 测试集: {test_matrix.nnz}条交互")
    
    # 设置训练矩阵
    dataset.train_matrix = train_matrix
    
    # 获取邻接矩阵
    ui_adj_matrices = dataset.get_adj_matrices()
    
    # 创建模型
    logger.info("创建模型...")
    model = SocialRecommender(
        num_users=dataset.num_users,
        num_items=dataset.num_items,
        embed_dim=args.embed_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        temperature=args.temperature
    ).to(device)
    
    # 打印模型参数
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型总参数: {total_params}")
    
    # 设置优化器
    logger.info("设置优化器...")
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # 设置学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, 
        patience=args.patience//2, verbose=True
    )
    
    # 开始训练
    logger.info("开始训练...")
    best_hr = 0
    best_epoch = 0
    no_improve = 0
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        model.train()
        total_loss = 0
        progress = tqdm(range(args.n_batch), desc=f"Epoch {epoch}")
        
        for _ in progress:
            batch_users = torch.randint(0, dataset.num_users, (args.batch_size,), device=device)
            loss = model.calculate_loss(batch_users, train_matrix, ui_adj_matrices)
            
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
            optimizer.step()
            
            total_loss += loss.item()
            progress.set_postfix({"loss": loss.item()})
        
        # 计算平均损失
        avg_loss = total_loss / args.n_batch
        logger.info(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/train", avg_loss, epoch)
        
        # 更新学习率
        scheduler.step(avg_loss)
        
        # 评估
        if epoch % args.eval_interval == 0:
            model.eval()
            hr, ndcg = evaluate(
                model, dataset, test_matrix, ui_adj_matrices,
                args.top_k, args.batch_size, device
            )
            
            logger.info(f"Epoch {epoch}/{args.epochs} - HR@{args.top_k}: {hr:.4f}, NDCG@{args.top_k}: {ndcg:.4f}")
            writer.add_scalar(f"HR@{args.top_k}/val", hr, epoch)
            writer.add_scalar(f"NDCG@{args.top_k}/val", ndcg, epoch)
            
            # 保存最佳模型
            if hr > best_hr:
                best_hr = hr
                best_epoch = epoch
                logger.info(f"发现更好的模型, HR: {hr:.4f}")
                no_improve = 0
                
                # 保存最佳模型
                best_model_path = os.path.join(args.output_dir, "best_model.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'hr': hr,
                    'ndcg': ndcg
                }, best_model_path)
                logger.info(f"保存最佳模型到 {best_model_path}")
                
                # 保存检查点
                checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'hr': hr,
                    'ndcg': ndcg
                }, checkpoint_path)
            else:
                no_improve += 1
                # 早停
                if no_improve >= args.patience:
                    logger.info(f"触发早停，训练结束")
                    break
    
    writer.close()
    logger.info(f"训练完成！最佳HR: {best_hr:.4f} (Epoch {best_epoch})")
    
    # 返回训练历史
    history = {
        'best_hr': best_hr,
        'best_epoch': best_epoch
    }
    return history


def _train_epoch(model: nn.Module, dataset: LastFMDataset, 
                ui_adj_matrices: Dict[str, torch.sparse.FloatTensor],
                optimizer: optim.Optimizer, args: argparse.Namespace) -> Tuple[float, float, float]:
    """单个训练轮次"""
    model.train()
    
    # 统计
    epoch_loss = 0.0
    epoch_bpr_loss = 0.0
    epoch_reg_loss = 0.0
    n_batches = 0
    
    # 训练循环
    for _ in range(args.n_batch):
        batch_data = dataset.get_batch_data(args.batch_size)
        
        if not batch_data:  # 空批次
            continue
            
        batch_data = batch_data[0]  # 取第一个批次
        
        # 提取数据
        users = batch_data['users']
        pos_items = batch_data['pos_items']
        neg_items = batch_data['neg_items']
        social_matrix = batch_data['social_matrix']
        
        # 前向传播
        batch_output = model(users, pos_items, neg_items, ui_adj_matrices, social_matrix)
        
        # 计算损失
        loss_dict = model.calculate_loss(batch_output)
        loss = loss_dict['total_loss']
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        # 更新参数
        optimizer.step()
        
        # 更新统计
        epoch_loss += loss.item()
        epoch_bpr_loss += loss_dict['bpr_loss'].item()
        epoch_reg_loss += loss_dict['reg_loss'].item()
        n_batches += 1
    
    # 计算平均损失
    if n_batches > 0:
        epoch_loss /= n_batches
        epoch_bpr_loss /= n_batches
        epoch_reg_loss /= n_batches
    
    return epoch_loss, epoch_bpr_loss, epoch_reg_loss


def evaluate(model: nn.Module, dataset: LastFMDataset, 
            test_matrix: Any, ui_adj_matrices: Dict[str, torch.sparse.FloatTensor],
            top_k: int, batch_size: int, device: torch.device) -> Tuple[float, float]:
    """
    评估模型性能
    
    参数:
        model: 社交推荐模型
        dataset: 数据集
        test_matrix: 测试集交互矩阵
        ui_adj_matrices: 用户-物品邻接矩阵
        top_k: 推荐列表长度
        batch_size: 批次大小
        device: 计算设备
        
    返回:
        hr: 平均命中率
        ndcg: 平均NDCG
    """
    model.eval()
    
    # 评估指标
    hit_ratio = []
    ndcg_list = []
    
    # 随机选择评估用户
    eval_users = np.random.choice(dataset.num_users, min(1000, dataset.num_users), replace=False)
    
    with torch.no_grad():
        for start_idx in range(0, len(eval_users), batch_size):
            end_idx = min(start_idx + batch_size, len(eval_users))
            batch_users = eval_users[start_idx:end_idx]
            batch_users_tensor = torch.LongTensor(batch_users).to(device)
            
            # 获取社交矩阵
            batch_data = dataset.get_batch_data(len(batch_users), train=False)
            if not batch_data:  # 空批次
                continue
                
            batch_data = batch_data[0]  # 取第一个批次
            social_matrix = batch_data['social_matrix']
            
            # 预测得分
            scores = model.predict(batch_users_tensor, ui_adj_matrices, social_matrix)
            
            # 如果返回的是字典，提取得分
            if isinstance(scores, dict):
                scores = scores.get('scores', scores)
                
            # 转换为NumPy数组
            scores = scores.cpu().numpy()
            
            # 屏蔽训练集中的物品
            for idx, user in enumerate(batch_users):
                # 获取训练物品
                train_items = dataset.train_matrix[user].nonzero()[1]
                scores[idx, train_items] = -np.inf
            
            # 计算每个用户的指标
            for idx, user in enumerate(batch_users):
                test_items = test_matrix[user].nonzero()[1]
                
                if len(test_items) == 0:
                    continue
                    
                # 获取推荐物品
                user_scores = scores[idx]
                recommended_items = np.argsort(-user_scores)[:top_k]
                
                # 计算指标
                hr, ndcg = calculate_metrics(recommended_items, test_items, top_k)
                hit_ratio.append(hr)
                ndcg_list.append(ndcg)
    
    # 计算平均指标
    if not hit_ratio:
        return 0.0, 0.0
        
    mean_hr = np.mean(hit_ratio)
    mean_ndcg = np.mean(ndcg_list)
    
    return mean_hr, mean_ndcg


def _save_model(model: nn.Module, optimizer: optim.Optimizer, epoch: int, 
                hr: float, ndcg: float, path: str) -> None:
    """保存模型"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'hr': hr,
        'ndcg': ndcg
    }, path)


def load_model(model: nn.Module, path: str, device: torch.device, 
              optimizer: Optional[optim.Optimizer] = None) -> Tuple[nn.Module, int, float]:
    """
    加载模型
    
    参数:
        model: 社交推荐模型
        path: 模型路径
        device: 计算设备
        optimizer: 优化器 (可选)
        
    返回:
        model: 加载后的模型
        epoch: 轮次
        hr: 命中率
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型文件 {path} 不存在")
        
    # 加载检查点
    checkpoint = torch.load(path, map_location=device)
    
    # 加载模型参数
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 如果提供了优化器，加载优化器参数
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    hr = checkpoint['hr']
    
    return model, epoch, hr


def main():
    # 参数解析
    parser = argparse.ArgumentParser(description='社交推荐系统训练')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data/lastfm',
                      help='数据集目录')
    parser.add_argument('--output_dir', type=str, default='./output',
                      help='输出目录')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                      help='测试集比例')
    
    # 模型参数
    parser.add_argument('--embed_dim', type=int, default=64,
                      help='嵌入维度')
    parser.add_argument('--n_layers', type=int, default=2,
                      help='图卷积层数')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Dropout率')
    parser.add_argument('--temperature', type=float, default=0.2,
                      help='Gumbel-Softmax温度参数')
    
    # 训练参数
    parser.add_argument('--lr', type=float, default=0.001,
                      help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                      help='权重衰减')
    parser.add_argument('--batch_size', type=int, default=1024,
                      help='批次大小')
    parser.add_argument('--n_batch', type=int, default=2000,
                      help='每轮训练的批次数')
    parser.add_argument('--epochs', type=int, default=200,
                      help='训练轮次')
    parser.add_argument('--patience', type=int, default=20,
                      help='早停耐心值')
    parser.add_argument('--eval_interval', type=int, default=1,
                      help='评估间隔')
    parser.add_argument('--top_k', type=int, default=10,
                      help='推荐列表长度')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                      help='梯度裁剪阈值')
    
    # 其他参数
    parser.add_argument('--gpu', type=int, default=0,
                      help='GPU ID，设为-1使用CPU')
    parser.add_argument('--seed', type=int, default=42,
                      help='随机种子')
    
    args = parser.parse_args()
    
    # 训练模型
    train_model(args)


if __name__ == '__main__':
    main() 