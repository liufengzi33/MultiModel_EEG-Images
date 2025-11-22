import torch
from utils.my_transforms import transform_cnn_2

class Config:
    def __init__(self):
        # 数据集参数
        self.subject_id = "01gh"
        # 通用训练参数
        self.batch_size = 16
        self.num_epochs = 4000
        self.learning_rate = 1e-3
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.factor = 0.5
        self.patience = 20
        self.alpha = 0.5
        self.beta = 0.5
        self.transform = transform_cnn_2

        # Early Stop 设置
        self.max_lr_plateaus = 8  # 最多衰减5次

        # 模型与路径设置
        self.base_eeg_model = 'EEGNet'
        self.base_image_model = 'AlexNet'
        self.image_model_type = "rsscnn"
        self.use_pretrained = True # 使用预训练的eeg&image模型
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'