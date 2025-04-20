import torch


class Config:
    def __init__(self):
        # 通用训练参数
        self.batch_size = 4
        self.num_epochs = 400
        self.learning_rate = 0.001
        self.momentum = 0.9

        # Early Stop 设置
        self.lr_decay_factor = 0.1
        self.max_lr_plateaus = 5  # 最多衰减5次

        # 模型与路径设置
        self.base_model_name = 'AlexNet'
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'

        # RSSCNN的超参数，网格搜索得到
        self.lambda_r = 0.1  # 可根据网格搜索结果手动修改
