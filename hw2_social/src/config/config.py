import torch

class Config:
    # 数据集配置
    DATASET = 'LastFM'  # 可选: 'LastFM', 'Flickr', 'Yelp'
    DATA_PATH = './data/'
    
    # 模型参数
    EMBEDDING_DIM = 50
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 2000
    NUM_EPOCHS = 200
    
    # 图卷积参数
    NUM_LAYERS = 2
    DROPOUT = 0.1
    
    # 社交影响参数
    NUM_IMPLICIT_NEIGHBORS = 30
    GUMBEL_TEMPERATURE = 0.1
    
    # 损失函数权重
    LAMBDA_VAE = 0.1
    LAMBDA_KL = 0.1
    
    # 评估参数
    TOP_K = [5, 10, 20]
    
    # 设备配置
    DEVICE = 'cpu'  # 使用CPU训练
    
    # 数据配置
    DATA_DIR = 'data/LastFM'
    INTERACTION_FILE = 'interactions.csv'
    SOCIAL_FILE = 'social.csv'
    
    # 社交图卷积配置
    SOCIAL_CONV_LAYERS = 2  # 社交图卷积层数
    
    # 评估配置
    EVAL_STEP = 5  # 每5个epoch评估一次
    
    # 早停配置
    PATIENCE = 20  # 早停耐心值
    MIN_DELTA = 0.0001  # 最小改善阈值 