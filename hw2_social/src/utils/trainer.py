import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.model.to(self.device)
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc='Training'):
            # 将数据移到设备
            user_ids = batch['user'].to(self.device)
            pos_items = batch['pos_item'].to(self.device)
            neg_items = batch['neg_items'][:, 0].to(self.device)  # 取第一个负样本
            
            # 构建邻接矩阵（简化版）
            batch_size = user_ids.size(0)
            
            # 前向传播（假设简化模型接口）
            scores, kl_losses = self.model(user_ids, torch.arange(self.model.item_embedding.weight.size(0)).to(self.device), None, None)
            
            # 计算损失 - BPR损失
            pos_scores = scores[torch.arange(batch_size), pos_items]
            neg_scores = scores[torch.arange(batch_size), neg_items]
            bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
            
            # 加上KL散度损失（如果有）
            if kl_losses:
                kl_loss = sum(kl_losses) / len(kl_losses)
                loss = bpr_loss + self.config.LAMBDA_KL * kl_loss
            else:
                loss = bpr_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
    
    def evaluate(self):
        self.model.eval()
        metrics = {k: 0.0 for k in self.config.TOP_K}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Evaluating'):
                # 将数据移到设备
                user_ids = batch['user'].to(self.device)
                pos_items = batch['pos_item'].to(self.device)
                
                # 获取预测分数
                scores, _ = self.model(user_ids, torch.arange(self.model.item_embedding.weight.size(0)).to(self.device), None, None)
                
                # 对每个用户计算top-k
                for k in self.config.TOP_K:
                    # 取top-k物品
                    _, top_items = torch.topk(scores, k)
                    
                    # 计算命中率
                    hits = [1 if pos_items[i] in top_items[i] else 0 for i in range(len(pos_items))]
                    metrics[k] += sum(hits) / len(hits)
        
        # 取平均
        for k in metrics:
            metrics[k] /= len(self.val_loader)
        
        return metrics
    
    def train(self, num_epochs):
        best_metric = 0
        best_model = None
        patience = self.config.PATIENCE
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # 训练一个epoch
            train_loss = self.train_epoch()
            
            # 评估
            metrics = self.evaluate()
            
            # 打印结果
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Validation Metrics:')
            for k, value in metrics.items():
                print(f'HR@{k}: {value:.4f}')
            
            # 早停检查
            current_metric = metrics[10]  # 使用HR@10作为早停指标
            if current_metric > best_metric + self.config.MIN_DELTA:
                best_metric = current_metric
                best_model = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        # 加载最佳模型
        if best_model is not None:
            self.model.load_state_dict(best_model)
        
        return best_metric
    