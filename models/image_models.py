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
            # TODO 使用 PlacesNet 作为特征提取器
            model = models.vgg16(pretrained=True)
            self.features = model.features
            self.out_dim = 512  # PlacesNet 特征提取器的输出维度
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
        return self.classifier(self.avg_pool(self.fusion_convs(combined)))

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
        self.rank = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1)
        )
        self._initialize_weights()

    def forward(self, x):
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
    def __init__(self, base_model_name):
        super(ImageFeatureExtractor, self).__init__()
        self.feature_extractor = BaseFeatureExtractor(base_model_name)
        self.fusion_convs = FusionNetwork(self.feature_extractor.out_dim).fusion_convs
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

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
    model = SSCNN(base_model_name="VGG")
    print(model)
    # 查看alexnet的输出维度
    model = models.alexnet(pretrained=True)
    print(model)
    # 测试模型
    model = RSSCNN(base_model_name="VGG")
    print(model)
