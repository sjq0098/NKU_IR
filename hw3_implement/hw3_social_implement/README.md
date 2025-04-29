# Lite-EIISRS 模型实现

本项目是Lite-EIISRS（轻量化嵌入和注意力推荐模型）的PyTorch实现，专门为LastFM数据集设计。

## 模型架构

Lite-EIISRS模型结合了以下几个关键组件：
- 矩阵分解(MF)和SVD初始化
- 统计特征提取和处理
- LightGCN用于用户-物品交互建模
- SocialGCN用于社交关系建模
- 多头注意力机制
- 残差MLP

## 环境要求

```
Python 3.7+
PyTorch 1.8.0+
```

详细依赖见`requirements.txt`。

## 安装

```bash
# 克隆代码库
git clone [repo-url]
cd [repo-dir]

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

1. **准备数据**

   确保LastFM数据集已放置在`data/hetrec2011-lastfm-2k`目录下。

2. **模型训练**

   ```bash
   python train.py --epochs 30 --embed_dim 64 --use_gpu
   ```

3. **参数说明**

   - `--data_dir`: 数据集路径，默认为'data/hetrec2011-lastfm-2k'
   - `--embed_dim`: 嵌入维度，默认为64
   - `--batch_size`: 批次大小，默认为1024
   - `--lr`: 学习率，默认为0.001
   - `--epochs`: 训练轮次，默认为30
   - `--weight_decay`: 权重衰减，默认为1e-5
   - `--n_layers_light`: LightGCN层数，默认为2
   - `--n_layers_social`: SocialGCN层数，默认为2
   - `--n_heads`: 注意力头数，默认为2
   - `--dropout`: Dropout率，默认为0.1
   - `--eval_interval`: 评估间隔，默认为1
   - `--top_k`: 推荐top-k物品，默认为10
   - `--seed`: 随机种子，默认为42
   - `--use_gpu`: 是否使用GPU
   - `--output_dir`: 输出目录，默认为'output'

4. **输出说明**

   训练结果会保存在`output`目录下，包括：
   - 模型权重文件 `model.pt`
   - 训练过程指标图表 `metrics_epoch_*.png`
   - 实验结果记录 `results.txt`

## 指标评估

模型使用以下指标进行评估：
- Hit Ratio@K (HR@K)
- Normalized Discounted Cumulative Gain@K (NDCG@K)
- Precision@K
- Recall@K 