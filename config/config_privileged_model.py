import os
from datetime import datetime

from utils.my_transforms import transform_cnn_2

class PrivilegedConfig:
    """特权学习网络配置类"""

    def __init__(self):
        # 模型参数
        self.eeg_model_name = 'EEGNetv1'
        self.image_model_name = 'PlacesNet'
        self.image_model_type = 'rsscnn'
        self.in_chans = 64
        self.n_classes = 2
        self.input_window_samples = 2000
        self.use_pretrained = True
        self.common_dim = 512
        self.private_dim = 256
        self.dropout_rate = 0.5
        self.alpha = 0.5  # 公共损失权重
        self.beta = 0.5  # 私有损失权重
        self.gamma = 0.3  # 蒸馏损失权重
        self.temperature = 2.0 # 蒸馏温度

        # 训练参数
        self.epochs = 100
        self.batch_size = 16
        self.lr = 1e-4
        self.momentum = 0.9
        self.weight_decay = 1e-5
        self.optimizer = 'adamw'
        self.scheduler = 'plateau'
        self.grad_clip = 1.0
        self.factor = 0.5
        self.patience = 10

        # 数据参数
        self.subject_id = '01gh'
        self.train_ratio = 0.8
        self.transform = transform_cnn_2

        # 输出设置
        self.output_dir = 'outputs/outputs_privileged'
        self.log_interval = 10
        self.save_interval = 10

        # 路径设置
        self.base_path = 'outputs'

        # 恢复训练
        self.resume_training = False
        self.checkpoint_path = ''

    def __post_init__(self):
        """初始化后自动创建输出目录结构"""
        # 创建基础输出目录
        os.makedirs(self.output_dir, exist_ok=True)

    def to_dict(self):
        """将配置转换为字典"""
        return {
            # 模型参数
            'eeg_model_name': self.eeg_model_name,
            'image_model_name': self.image_model_name,
            'image_model_type': self.image_model_type,
            'in_chans': self.in_chans,
            'n_classes': self.n_classes,
            'input_window_samples': self.input_window_samples,
            'use_pretrained': self.use_pretrained,
            'common_dim': self.common_dim,
            'private_dim': self.private_dim,
            'dropout_rate': self.dropout_rate,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'temperature': self.temperature,

            # 训练参数
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'momentum': self.momentum,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'grad_clip': self.grad_clip,
            'factor': self.factor,
            'patience': self.patience,

            # 数据参数
            'subject_id': self.subject_id,
            'train_ratio': self.train_ratio,
            'transform': self.transform,


            # 输出设置
            'output_dir': self.output_dir,
            'log_interval': self.log_interval,
            'save_interval': self.save_interval,

            # 路径设置
            'base_path': self.base_path,

            # 恢复训练
            'resume_training': self.resume_training,
            'checkpoint_path': self.checkpoint_path
        }

    def get_output_dir(self):
        """动态生成输出目录路径"""
        model_subdir = f"{self.eeg_model_name}_{self.image_model_name}_{self.image_model_type}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.output_dir, model_subdir, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

# 预定义配置
def get_default_config():
    """获取默认配置"""
    return PrivilegedConfig()


def get_debug_config():
    """获取调试配置（快速测试用）"""
    config = PrivilegedConfig()
    config.epochs = 5
    config.batch_size = 4
    config.log_interval = 2
    config.save_interval = 2
    config.output_dir = 'outputs/debug_privileged'
    return config


def get_large_config():
    """获取大型模型配置"""
    config = PrivilegedConfig()
    config.common_dim = 1024
    config.private_dim = 512
    config.batch_size = 32
    config.lr = 5e-5
    return config


def get_small_config():
    """获取小型模型配置"""
    config = PrivilegedConfig()
    config.common_dim = 256
    config.private_dim = 128
    config.batch_size = 8
    config.lr = 2e-4
    return config


# 配置字典映射
CONFIG_REGISTRY = {
    'default': get_default_config,
    'debug': get_debug_config,
    'large': get_large_config,
    'small': get_small_config,
}


def get_config(config_name='default'):
    """通过名称获取配置"""
    if config_name in CONFIG_REGISTRY:
        return CONFIG_REGISTRY[config_name]()
    else:
        print(f"警告: 配置 '{config_name}' 不存在，使用默认配置")
        return get_default_config()