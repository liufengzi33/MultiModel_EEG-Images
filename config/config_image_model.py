import torch
from utils.my_transforms import transform_cnn_2

class Config:
    def __init__(self):
        # 通用训练参数
        self.batch_size = 4
        self.num_epochs = 4000
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.factor = 0.1
        self.patience = 5
        self.transform = transform_cnn_2

        # Early Stop 设置
        self.max_lr_plateaus = 5  # 最多衰减5次

        # 模型与路径设置
        self.base_model_name = 'VGG'  # AlexNet, VGG, PlacesNet
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'

        # RSSCNN的超参数，网格搜索得到
        self.lambda_r = 0.5  # 可根据网格搜索结果手动修改
