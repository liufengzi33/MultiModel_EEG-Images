import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.nn.init as init


class BaseFeatureExtractor(nn.Module):
    def __init__(self, base_model_name):
        super(BaseFeatureExtractor, self).__init__()

        if base_model_name == "AlexNet":
            model = models.alexnet(pretrained=True)
            self.features = model.features
            self.out_dim = 256  # Alexnet 特征提取器的输出维度
        elif base_model_name == "VGG":
            model = models.vgg19(pretrained=True)
            self.features = model.features
            self.out_dim = 512  # VGG19 特征提取器的输出维度
        elif base_model_name == "PlacesNet":
            # 使用 PlacesNet 作为特征提取器
            model = models.alexnet(num_classes=365)

            try:
                # 加载完整 checkpoint
                weights_path = os.path.join('weights/alexnet_places365.pth.tar')
                checkpoint = torch.load(weights_path, map_location='cpu')
                # 提取 state_dict
                state_dict = checkpoint['state_dict']
                # 去掉 'module.' 前缀
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace('module.', '')  # 去除多GPU训练时的 module. 前缀
                    new_state_dict[name] = v
                # 加载权重
                model.load_state_dict(new_state_dict)
                print("✅ Successfully loaded PlacesNet pretrained weights")
            except Exception as e:
                print(f"❌ Failed to load pretrained weights: {e}")
                print("⚠️ Will use randomly initialized weights instead")

            self.features = model.features
            self.out_dim = 256  # PlacesNet 特征提取器的输出维度
        else:
            raise ValueError("Unsupported base model. Choose from 'AlexNet', 'VGG', or 'PlacesNet'")

    def forward(self, x):
        x = self.features(x)
        return x


class FusionNetwork(nn.Module):
    def __init__(self, input_dim):
        super(FusionNetwork, self).__init__()
        self.fusion_convs = nn.Sequential(
            nn.Conv2d(input_dim * 2, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            # TODO 是不是加上归一化层更好呢
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),  # 可选，增加模型的鲁棒性
            nn.Linear(256, 2),  # 二分类
        )
        self._initialize_weights()

    def forward(self, x1, x2):
        combined = torch.cat([x1, x2], dim=1)
        features = self.fusion_convs(combined)
        pooled = self.avg_pool(features)
        flattened = pooled.view(pooled.size(0), -1)
        return self.classifier(flattened)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用高斯初始化（均值0，标准差0.01）
                init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0.1)  # 初始化偏置
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0.1)  # 初始化偏置


class RankingNetwork(nn.Module):
    def __init__(self, input_dim):
        super(RankingNetwork, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.rank = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1)
        )
        self._initialize_weights()

    def forward(self, x):
        # 先进行全局平均池化
        x = self.avg_pool(x)
        # 展平
        x = x.view(x.size(0), -1)
        return self.rank(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0.1)


class ImageFeatureExtractor(nn.Module):
    """
    封装特征提取器和融合网络，当作后续我的多模态模型的特征提取器
    """

    def __init__(self, base_model_name, pretrained_rsscnn=None):
        super(ImageFeatureExtractor, self).__init__()
        self.feature_extractor = BaseFeatureExtractor(base_model_name)
        self.fusion_convs = FusionNetwork(self.feature_extractor.out_dim).fusion_convs
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 添加 out_dim 属性
        self.out_dim = 512  # FusionNetwork.fusion_convs 的输出通道数

        if pretrained_rsscnn is not None:
            # 加载预训练模型的权重
            self.load_from_rsscnn(pretrained_rsscnn)
            print("  ✅ 成功加载了图像通路初始化权重")

    def load_from_rsscnn(self, rsscnn_model):
        # 加载特征提取器权重
        self.feature_extractor.load_state_dict(rsscnn_model.feature_extractor.state_dict())
        # 加载融合卷积权重
        self.fusion_convs.load_state_dict(rsscnn_model.fusion.fusion_convs.state_dict())

    def forward(self, x1, x2):
        f1 = self.feature_extractor(x1)
        f2 = self.feature_extractor(x2)
        fused = torch.cat([f1, f2], dim=1)
        conv_features = self.fusion_convs(fused)
        pooled = self.avg_pool(conv_features)
        return pooled.view(pooled.size(0), -1)


class SSCNN(nn.Module):
    def __init__(self, base_model_name):
        super(SSCNN, self).__init__()
        self.feature_extractor = BaseFeatureExtractor(base_model_name)
        self.fusion = FusionNetwork(self.feature_extractor.out_dim)

    def forward(self, x1, x2):
        f1 = self.feature_extractor(x1)
        f2 = self.feature_extractor(x2)
        return self.fusion(f1, f2)


class RSSCNN(nn.Module):
    def __init__(self, base_model_name, lambda_r=0.1):
        super(RSSCNN, self).__init__()
        self.feature_extractor = BaseFeatureExtractor(base_model_name)
        self.fusion = FusionNetwork(self.feature_extractor.out_dim)
        self.ranking = RankingNetwork(self.feature_extractor.out_dim)
        self.lambda_r = lambda_r
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x1, x2):
        f1 = self.feature_extractor(x1)
        f2 = self.feature_extractor(x2)
        class_out = self.fusion(f1, f2)
        rank1 = self.ranking(f1)
        rank2 = self.ranking(f2)
        return class_out, rank1, rank2

    def compute_loss(self, class_out, rank1, rank2, labels):
        classification_loss = self.loss_fn(class_out, labels)
        # 标签的转换0/1 -> -1/1
        y = labels * 2 - 1
        ranking_loss = torch.mean(F.relu(y.view(-1, 1) * (rank2 - rank1)) ** 2)
        return classification_loss + self.lambda_r * ranking_loss


if __name__ == "__main__":
    # 测试模型
    model = SSCNN(base_model_name="PlacesNet")
    print(model)
    # 查看alexnet的输出维度
    model = models.alexnet(pretrained=True)
    print(model)
    # 测试模型
    model = RSSCNN(base_model_name="PlacesNet")
    print(model)

    # 多模态图像特征提取器的使用示例
    rsscnn = RSSCNN(base_model_name="PlacesNet")
    rsscnn.load_state_dict(torch.load("rsscnn_weights.pth"))

    image_extractor = ImageFeatureExtractor(base_model_name="PlacesNet", pretrained_rsscnn=rsscnn)
