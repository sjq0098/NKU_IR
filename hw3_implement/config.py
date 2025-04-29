from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    embed_dim: int = 64
    n_layers: int = 2
    dropout: float = 0.1
    temperature: float = 0.2

@dataclass
class TrainingConfig:
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    num_epochs: int = 100
    early_stopping_patience: int = 10
    device: str = "cuda"  # or "cpu"
    num_workers: int = 4
    save_dir: str = "checkpoints"
    log_dir: str = "logs"

@dataclass
class DataConfig:
    data_path: str = "data"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42
    num_negatives: int = 4  # 每个正样本对应的负样本数量

@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    
    def __post_init__(self):
        # 确保目录存在
        import os
        os.makedirs(self.training.save_dir, exist_ok=True)
        os.makedirs(self.training.log_dir, exist_ok=True) 