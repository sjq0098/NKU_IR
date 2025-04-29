import os
import logging
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_dir: str, experiment_name: str = None):
        self.log_dir = log_dir
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_path = os.path.join(log_dir, self.experiment_name)
        
        # 创建日志目录
        os.makedirs(self.log_path, exist_ok=True)
        
        # 设置日志记录器
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # 文件处理器
        file_handler = logging.FileHandler(os.path.join(self.log_path, "training.log"))
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(console_handler)
        
        # TensorBoard写入器
        self.writer = SummaryWriter(self.log_path)
        
    def log_metrics(self, metrics: dict, step: int):
        """记录指标到TensorBoard"""
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)
            
    def log_model(self, model: torch.nn.Module, input_shape: tuple):
        """记录模型结构到TensorBoard"""
        self.writer.add_graph(model, torch.randn(input_shape))
        
    def log_text(self, text: str):
        """记录文本信息"""
        self.logger.info(text)
        
    def log_hyperparameters(self, config: dict):
        """记录超参数"""
        self.logger.info("Hyperparameters:")
        for key, value in config.items():
            self.logger.info(f"{key}: {value}")
            
    def close(self):
        """关闭所有记录器"""
        self.writer.close()
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler) 