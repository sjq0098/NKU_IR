# 社交推荐系统

基于图神经网络的社交推荐系统，使用LastFM数据集进行训练和评估。

## 项目结构

```
.
├── config.py           # 配置文件
├── logger.py           # 日志记录工具
├── main.py             # 主程序入口
├── social_recommender.py # 社交推荐模型
├── train.py            # 训练脚本
├── utils/              # 工具函数
│   ├── data_loader.py  # 数据加载工具
│   └── data_utils.py   # 数据处理工具
├── data/               # 数据目录
│   └── hetrec2011-lastfm-2k/ # LastFM数据集
├── checkpoints/        # 模型检查点保存目录
└── logs/               # 日志保存目录
```

## 环境配置

```bash
# 安装依赖
pip install -r requirements.txt
```

## 运行方法

```bash
# 运行主程序
python main.py
```

## 模型参数

在 `main.py` 中可以调整以下参数：

```python
# 模型参数
config.model.embed_dim = 64      # 嵌入维度
config.model.n_layers = 2        # 图卷积层数
config.model.dropout = 0.1       # Dropout比例
config.model.temperature = 0.2   # 温度参数

# 训练参数
config.training.batch_size = 256 # 批次大小
config.training.learning_rate = 0.001  # 学习率
config.training.weight_decay = 1e-5    # 权重衰减
config.training.num_epochs = 100       # 训练周期数
config.training.early_stopping_patience = 10  # 早停耐心值

# 数据参数
config.data.num_negatives = 4    # 负采样数量
```

## 可视化

训练过程中的指标会记录到TensorBoard中，可以使用以下命令查看：

```bash
tensorboard --logdir=logs
```

## 评估指标

- HR@5: 前5个推荐中命中率
- HR@10: 前10个推荐中命中率
- NDCG@5: 前5个推荐的归一化折扣累积增益
- NDCG@10: 前10个推荐的归一化折扣累积增益 