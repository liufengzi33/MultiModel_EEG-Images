import torch
import torch.nn as nn
import os
from utils.model_loader import ModelLoader


class MultiModalFusionNetwork(nn.Module):
    def __init__(self,
                 eeg_model_name="EEGNet",
                 image_model_name="PlacesNet",
                 image_model_type="rsscnn",  # "rsscnn" 或 "sscnn"
                 in_chans=64,
                 n_classes=2,
                 input_window_samples=2000,
                 use_pretrained_eeg=True,
                 use_pretrained_image=True,
                 base_path="outputs",
                 fusion_dim=512,
                 dropout_rate=0.5,
                 fusion_method="concatenate"):
        """
        多模态融合网络

        Args:
            eeg_model_name: EEG模型名称 ("EEGNet" 或 "ShallowFBCSPNet")
            image_model_name: 图像基础模型名称 ("AlexNet", "VGG", 或 "PlacesNet")
            image_model_type: 图像模型类型 ("rsscnn" 或 "sscnn")
            in_chans: EEG输入通道数
            n_classes: 分类类别数
            input_window_samples: EEG输入时间点数
            use_pretrained_eeg: 是否使用预训练的EEG模型
            use_pretrained_image: 是否使用预训练的图像模型
            base_path: 模型保存的基础路径
            fusion_dim: 融合特征维度
            dropout_rate: dropout率
            fusion_method: 融合方法 ("concatenate", "add", "weighted")
        """
        super(MultiModalFusionNetwork, self).__init__()

        # 初始化模型加载器
        self.model_loader = ModelLoader(base_path)

        # 加载预训练模型
        pretrained_ssbcinet = None
        pretrained_image_model = None

        if use_pretrained_eeg:
            pretrained_ssbcinet = self.model_loader.load_eeg_model(
                model_name=eeg_model_name,
                in_chans=in_chans,
                n_classes=n_classes,
                input_window_samples=input_window_samples
            )

        if use_pretrained_image:
            pretrained_image_model = self.model_loader.load_image_model(
                model_type=image_model_type,
                base_model_name=image_model_name
            )

        # 初始化EEG特征提取通路
        self.eeg_feature_net = self._build_eeg_path(
            eeg_model_name=eeg_model_name,
            in_chans=in_chans,
            n_classes=n_classes,
            input_window_samples=input_window_samples,
            pretrained_ssbcinet=pretrained_ssbcinet
        )

        # 初始化图像特征提取通路
        self.image_feature_net = self._build_image_path(
            image_model_name=image_model_name,
            pretrained_image_model=pretrained_image_model,
            image_model_type=image_model_type
        )

        # 特征投影层
        self.eeg_projection = nn.Sequential(
            nn.Linear(self.eeg_feature_net.out_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.image_projection = nn.Sequential(
            nn.Linear(self.image_feature_net.out_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 多模态融合配置
        self.fusion_method = fusion_method

        if self.fusion_method == "concatenate":
            fusion_input_dim = fusion_dim * 2
        elif self.fusion_method == "add":
            fusion_input_dim = fusion_dim
        elif self.fusion_method == "weighted":
            self.attention_fusion = CrossModalAttention(fusion_dim)
            fusion_input_dim = fusion_dim
        else:
            raise ValueError(f"不支持的融合方法: {fusion_method}")

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(128, n_classes if n_classes > 2 else 1)
        )

        # 初始化权重（只初始化新添加的层）
        self._initialize_weights()

    def _build_eeg_path(self, eeg_model_name, in_chans, n_classes, input_window_samples, pretrained_ssbcinet):
        """构建EEG特征提取通路"""
        from eeg_models import EEGFeatureExtractor, EEGFusionNetwork

        class EEGFeaturePath(nn.Module):
            def __init__(self, feature_extractor, fusion_net, out_dim):
                super(EEGFeaturePath, self).__init__()
                self.feature_extractor = feature_extractor
                self.fusion = fusion_net
                self.out_dim = out_dim

            def forward(self, x1, x2):
                f1 = self.feature_extractor(x1)
                f2 = self.feature_extractor(x2)
                fused = self.fusion(f1, f2)
                return fused

        if pretrained_ssbcinet is not None:
            # 使用预训练的SSBCINet
            print("✅ 使用预训练的SSBCINet初始化EEG通路")
            feature_extractor = pretrained_ssbcinet.feature_extractor
            fusion_net = pretrained_ssbcinet.fusion
            out_dim = 512  # SSBCINet fusion输出维度
            print("  ✅ 成功加载了脑电通路初始化权重")
        else:
            # 随机初始化
            print("🔄 随机初始化EEG通路")
            feature_extractor = EEGFeatureExtractor(
                model_name=eeg_model_name,
                in_chans=in_chans,
                n_classes=n_classes,
                input_window_samples=input_window_samples,
            )
            fusion_net = EEGFusionNetwork(feature_extractor.out_dim)
            out_dim = 512  # EEGFusionNetwork输出维度

        return EEGFeaturePath(feature_extractor, fusion_net, out_dim)

    def _build_image_path(self, image_model_name, pretrained_image_model, image_model_type):
        """构建图像特征提取通路"""
        from image_models import ImageFeatureExtractor

        if pretrained_image_model is not None:
            # 使用预训练的图像模型
            print(f"✅ 使用预训练的{image_model_type.upper()}初始化图像通路")
            return ImageFeatureExtractor(
                base_model_name=image_model_name,
                pretrained_rsscnn=pretrained_image_model
            )
        else:
            # 随机初始化
            print("🔄 随机初始化图像通路")
            return ImageFeatureExtractor(
                base_model_name=image_model_name,
                pretrained_rsscnn=None
            )

    def forward(self, eeg1, eeg2, img1, img2):
        """
        前向传播
        """
        # 提取EEG特征
        eeg_features = self.eeg_feature_net(eeg1, eeg2)
        eeg_features = self.eeg_projection(eeg_features)

        # 提取图像特征
        image_features = self.image_feature_net(img1, img2)
        image_features = self.image_projection(image_features)

        # 多模态融合
        if self.fusion_method == "concatenate":
            fused_features = torch.cat([eeg_features, image_features], dim=1)
        elif self.fusion_method == "add":
            fused_features = eeg_features + image_features
        elif self.fusion_method == "weighted":
            fused_features = self.attention_fusion(eeg_features, image_features)

        # 分类
        logits = self.classifier(fused_features)

        if logits.shape[1] == 1:
            return logits.squeeze()  # 二分类
        else:
            return logits  # 多分类

    def _initialize_weights(self):
        """只初始化新添加的层（投影层和分类器）"""
        for m in self.modules():
            if (isinstance(m, nn.Linear) and
                    m in [layer for layer in self.eeg_projection.modules()] +
                    [layer for layer in self.image_projection.modules()] +
                    [layer for layer in self.classifier.modules()]):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)


class CrossModalAttention(nn.Module):
    """跨模态注意力融合模块"""

    def __init__(self, feature_dim):
        super(CrossModalAttention, self).__init__()
        self.feature_dim = feature_dim

        # 注意力权重计算
        self.eeg_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )

        self.image_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, eeg_features, image_features):
        # 计算注意力权重
        eeg_weights = self.eeg_attention(eeg_features)
        image_weights = self.image_attention(image_features)

        # 归一化权重
        total_weights = eeg_weights + image_weights + 1e-8
        eeg_weights = eeg_weights / total_weights
        image_weights = image_weights / total_weights

        # 加权融合
        fused_features = eeg_weights * eeg_features + image_weights * image_features

        return fused_features


if __name__ == "__main__":

    # 测试模型加载和初始化

    print("=== 检查可用模型 ===")
    loader = ModelLoader()
    available = loader.get_available_models()
    print("可用EEG模型:", available["eeg_models"])
    print("可用RSSCNN模型:", available["image_models"]["rsscnn"])
    print("可用SSCNN模型:", available["image_models"]["sscnn"])

    print("\n=== 测试多模态网络 ===")

    # 测试1: 使用预训练模型（如果存在）
    if available["eeg_models"] and available["image_models"]["rsscnn"]:
        eeg_model = available["eeg_models"][0]
        image_model = available["image_models"]["rsscnn"][0]

        print(f"使用 {eeg_model} + RSSCNN-{image_model}")

        model = MultiModalFusionNetwork(
            eeg_model_name=eeg_model,
            image_model_name=image_model,
            image_model_type="rsscnn",
            use_pretrained_eeg=True,
            use_pretrained_image=True
        )

        model_sscnn = MultiModalFusionNetwork(
            eeg_model_name=eeg_model,
            image_model_name=image_model,
            image_model_type="sscnn",
            use_pretrained_eeg=True,
            use_pretrained_image=True
        )

        # 测试前向传播
        eeg1 = torch.randn(2, 64, 2000)
        eeg2 = torch.randn(2, 64, 2000)
        img1 = torch.randn(2, 3, 224, 224)
        img2 = torch.randn(2, 3, 224, 224)

        output = model(eeg1, eeg2, img1, img2)
        print(f"输出形状: {output.shape}")

    # 测试2: 随机初始化
    print("\n=== 测试随机初始化 ===")
    model_random = MultiModalFusionNetwork(
        use_pretrained_eeg=False,
        use_pretrained_image=False
    )
    output_random = model_random(eeg1, eeg2, img1, img2)
    print(f"随机初始化输出形状: {output_random.shape}")