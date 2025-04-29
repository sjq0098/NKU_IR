import os
import argparse
import torch
import numpy as np
import logging
import torch.optim as optim
from tqdm import tqdm

from utils import set_seed, load_data, split_data, create_normalized_adj_matrix, \
                  create_normalized_social_matrix, convert_sparse_matrix_to_sparse_tensor, \
                  SVDFeatureExtractor, evaluate, logger
from model import SocialRecommender

def train_one_epoch(model, optimizer, train_user_items, args, device):
    """训练一个epoch"""
    model.train()
    
    user_indices = list(train_user_items.keys())
    np.random.shuffle(user_indices)
    
    total_loss = 0.0
    total_bpr_loss = 0.0
    total_infonce_loss = 0.0
    
    num_batches = (len(user_indices) + args.batch_size - 1) // args.batch_size
    
    # 移除在epoch开始时的模型前向传播
    # user_embeds, item_embeds = model() 
    
    for i in tqdm(range(num_batches), desc="训练中", leave=False, ncols=80):
        # 获取批次用户
        batch_users = user_indices[i * args.batch_size: (i + 1) * args.batch_size]
        
        # --- 在每个batch内部执行模型前向传播 --- 
        # 这会为每个batch重新计算嵌入，并创建新的计算图
        user_embeds, item_embeds = model()
        # --- 
        
        pos_items = []
        neg_items_list = []
        
        for user in batch_users:
            # 正样本采样
            pos_item = np.random.choice(train_user_items[user])
            pos_items.append(pos_item)
            
            # 负样本采样
            neg_items = []
            for _ in range(args.num_negatives):
                while True:
                    neg_item = np.random.randint(0, model.num_items)
                    if neg_item not in train_user_items.get(user, []):
                        break
                neg_items.append(neg_item)
            neg_items_list.append(neg_items)
        
        batch_users_tensor = torch.LongTensor(batch_users).to(device)
        batch_pos_items_tensor = torch.LongTensor(pos_items).to(device)
        batch_neg_items_tensor = torch.LongTensor(neg_items_list).to(device)
        
        # 计算BPR损失
        bpr_losses = []
        for j in range(args.num_negatives):
            neg_j = batch_neg_items_tensor[:, j]
            # 注意：损失函数现在使用当前batch计算出的embeddings
            bpr_loss_j = model.bpr_loss(
                batch_users_tensor, batch_pos_items_tensor, neg_j, user_embeds, item_embeds
            )
            bpr_losses.append(bpr_loss_j)
        # 避免空列表求和/除零
        bpr_loss = sum(bpr_losses) / len(bpr_losses) if bpr_losses else torch.tensor(0.0).to(device)

        # 计算InfoNCE损失
        infonce_loss = model.infonce_loss(
            batch_users_tensor, batch_pos_items_tensor, user_embeds, item_embeds
        )
        
        # 计算总损失
        loss = args.bpr_weight * bpr_loss + args.infonce_weight * infonce_loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward() # 现在对每个batch独立的计算图进行反向传播
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        
        # 累计损失
        total_loss += loss.item()
        total_bpr_loss += bpr_loss.item()
        total_infonce_loss += infonce_loss.item()
        
    avg_loss = total_loss / num_batches
    avg_bpr_loss = total_bpr_loss / num_batches
    avg_infonce_loss = total_infonce_loss / num_batches
    
    return avg_loss, avg_bpr_loss, avg_infonce_loss

def main():
    """主函数"""
    # --- 参数解析 ---
    parser = argparse.ArgumentParser(description='运行社交推荐模型')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='./data/hetrec2011-lastfm-2k/user_artists.dat',
                      help='用户-物品交互数据路径')
    parser.add_argument('--social_path', type=str, default='./data/hetrec2011-lastfm-2k/user_friends.dat',
                      help='社交关系数据路径')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例')
    
    # 模型参数
    parser.add_argument('--embed_dim', type=int, default=64, help='嵌入维度')
    parser.add_argument('--n_layers', type=int, default=3, help='用户-物品图卷积层数')
    parser.add_argument('--social_layers', type=int, default=2, help='社交图卷积层数')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout概率')
    parser.add_argument('--temperature', type=float, default=0.07, help='InfoNCE损失温度参数')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=1024, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减 (L2正则化)')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪阈值')
    parser.add_argument('--num_negatives', type=int, default=4, help='每个正样本的负样本数量')
    parser.add_argument('--patience', type=int, default=15, help='早停耐心值')
    
    # 损失权重
    parser.add_argument('--bpr_weight', type=float, default=1.0, help='BPR损失权重')
    parser.add_argument('--infonce_weight', type=float, default=0.5, help='InfoNCE损失权重')
    
    # 评估参数
    parser.add_argument('--topk_list', type=int, nargs='+', default=[10, 20], help='推荐物品数量列表')
    parser.add_argument('--eval_interval', type=int, default=5, help='评估间隔 (epochs)')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    parser.add_argument('--use_svd', action='store_true', default=True, help='是否使用SVD初始化嵌入')

    args = parser.parse_args()
    
    # --- 设置与初始化 ---
    set_seed(args.seed)
    device = torch.device(args.device)
    logger.info(f"使用设备: {device}")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --- 数据加载与预处理 ---
    R_all, S_all, user_items_dict_all, num_users, num_items = load_data(args.data_path, args.social_path)
    
    R_train_csr, train_user_items, val_user_items, test_user_items = split_data(
        user_items_dict_all, R_all.shape, args.test_ratio, args.val_ratio, args.seed
    )
    
    R_norm, R_item_norm, R_bi_norm = create_normalized_adj_matrix(R_train_csr)
    S_norm = create_normalized_social_matrix(S_all)
    
    # 转换为Tensor
    R_norm_tensor = convert_sparse_matrix_to_sparse_tensor(R_norm).to(device)
    R_item_norm_tensor = convert_sparse_matrix_to_sparse_tensor(R_item_norm).to(device)
    # R_bi_norm_tensor = convert_sparse_matrix_to_sparse_tensor(R_bi_norm).to(device) # 如果模型需要，则取消注释
    S_norm_tensor = convert_sparse_matrix_to_sparse_tensor(S_norm).to(device) if S_norm is not None else None

    # SVD 特征
    user_svd_tensor, item_svd_tensor = None, None
    if args.use_svd:
        svd_extractor = SVDFeatureExtractor(k_components=args.embed_dim)
        user_svd, item_svd = svd_extractor.fit_transform(R_train_csr)
        user_svd_tensor = torch.FloatTensor(user_svd).to(device)
        item_svd_tensor = torch.FloatTensor(item_svd).to(device)

    # --- 模型创建 ---
    model = SocialRecommender(
        num_users=num_users,
        num_items=num_items,
        embed_dim=args.embed_dim,
        n_layers=args.n_layers,
        social_layers=args.social_layers,
        dropout=args.dropout,
        temperature=args.temperature,
        device=device
    ).to(device)
    
    # 将预计算的矩阵和SVD特征设置到模型中
    model.R_norm_sp = R_norm_tensor
    model.R_item_norm_sp = R_item_norm_tensor
    model.S_norm_sp = S_norm_tensor
    model.user_svd = user_svd_tensor
    model.item_svd = item_svd_tensor
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型可训练参数: {total_params:,}")

    # --- 训练设置 ---
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    best_hr_val = 0.0
    best_epoch = 0
    patience_counter = 0

    # --- 训练循环 ---
    logger.info("开始训练...")
    for epoch in range(args.epochs):
        avg_loss, avg_bpr_loss, avg_infonce_loss = train_one_epoch(
            model, optimizer, train_user_items, args, device
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"周期 {epoch+1}/{args.epochs} - "
                    f"损失: {avg_loss:.4f}, BPR: {avg_bpr_loss:.4f}, InfoNCE: {avg_infonce_loss:.4f}, "
                    f"学习率: {current_lr:.6f}")
        
        scheduler.step()
        
        # --- 评估与早停 ---
        if (epoch + 1) % args.eval_interval == 0:
            logger.info(f"开始评估周期 {epoch+1}...")
            # 获取当前嵌入用于评估
            with torch.no_grad():
                 user_embeds_eval, item_embeds_eval = model()
            
            val_metrics = evaluate(
                model, val_user_items, user_embeds_eval, item_embeds_eval, 
                topk_list=args.topk_list, device=device
            )
            
            # 打印验证集指标 (以 HR@10 为例)
            hr_10_val = val_metrics.get(f'HR@{args.topk_list[0]}', 0.0)
            ndcg_10_val = val_metrics.get(f'NDCG@{args.topk_list[0]}', 0.0)
            logger.info(f"验证集评估 - HR@{args.topk_list[0]}: {hr_10_val:.4f}, NDCG@{args.topk_list[0]}: {ndcg_10_val:.4f}")
            
            # 早停逻辑
            if hr_10_val > best_hr_val:
                best_hr_val = hr_10_val
                best_epoch = epoch + 1
                patience_counter = 0
                # 保存最佳模型
                model_save_path = os.path.join(args.save_dir, 'best_model.pth')
                torch.save(model.state_dict(), model_save_path)
                logger.info(f"保存最佳模型 (HR@{args.topk_list[0]}={best_hr_val:.4f}) 到 {model_save_path}")
            else:
                patience_counter += 1
                logger.info(f"验证集性能未提升, 耐心计数: {patience_counter}/{args.patience}")
                if patience_counter >= args.patience:
                    logger.info(f"早停触发: {args.patience} 轮未见改善")
                    break

    # --- 最终测试 ---
    logger.info("训练结束，加载最佳模型进行最终测试...")
    best_model_path = os.path.join(args.save_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        logger.info(f"从 {best_model_path} 加载最佳模型 (周期 {best_epoch})。")
        
        with torch.no_grad():
            user_embeds_test, item_embeds_test = model()
        
        test_metrics = evaluate(
            model, test_user_items, user_embeds_test, item_embeds_test, 
            topk_list=args.topk_list, device=device
        )
        
        logger.info("--- 最终测试结果 ---")
        for k in args.topk_list:
            logger.info(f"@{k}: HR={test_metrics.get(f'HR@{k}', 0.0):.4f}, "
                      f"NDCG={test_metrics.get(f'NDCG@{k}', 0.0):.4f}, "
                      f"Precision={test_metrics.get(f'Precision@{k}', 0.0):.4f}, "
                      f"Recall={test_metrics.get(f'Recall@{k}', 0.0):.4f}")
    else:
        logger.warning("未找到最佳模型文件，无法进行最终测试。")

    logger.info("运行完成!")

if __name__ == '__main__':
    main() 