import torch
from utils.my_transforms import transform_cnn_2

class Config:
    def __init__(self):
        # 通用训练参数
        self.batch_size = 4
        self.num_epochs = 400
        self.learning_rate = 1e-3
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.factor = 0.1
        self.patience = 10
        self.transform = transform_cnn_2

        # Early Stop 设置
        self.max_lr_plateaus = 5  # 最多衰减5次

        # 模型与路径设置
        self.base_model_name = 'PlacesNet'  # AlexNet, VGG, PlacesNet
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # RSSCNN的超参数，网格搜索得到
        self.lambda_r = 0.2  # 可根据网格搜索结果手动修改 AlexNet: 0.5, VGG: 0.2, PlacesNet: 0.5
