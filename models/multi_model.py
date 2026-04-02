import torch
import torch.nn as nn
import torch.nn.functional as F
from models.eeg_models import EEGFeatureExtractor, EEGFusionNetwork
from utils.model_loader import ModelLoader
from models.image_models import ImageFeatureExtractor


class MultiModalFusionNetwork(nn.Module):
    def __init__(self,
                 eeg_model_name="EEGNetv1",
                 image_model_name="PlacesNet",
                 image_model_type="rsscnn",
                 subject_id="01gh",  # <--- 新增：接收被试ID
                 in_chans=64,
                 n_classes=2,
                 input_window_samples=2000,
                 use_pretrained_eeg=True,
                 use_pretrained_image=True,
                 base_path="outputs",
                 common_dim=512,
                 private_dim=256,
                 dropout_rate=0.5,
                 alpha=0.5,  # 公共损失权重
                 beta=0.5, # 私有损失权重
                 ablation_mode="none"):
        """
        改进的多模态融合网络 - 基于脑机耦合学习思想

        Args:
            common_dim: 公共特征维度
            private_dim: 私有特征维度
            alpha: 公共通道相似性损失权重
            beta: 私有通道差异性损失权重
            ablation_mode 消融实验模式:
            - "none": 完整模型 (默认)
            - "baseline_concat": 仅拼接特征 (Baseline)
            - "no_cmd": 去掉公共通道相似性损失 (CMD)
            - "no_ortho": 去掉私有通道差异性损失 (Orthogonality)
        """
        super(MultiModalFusionNetwork, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.ablation_mode = ablation_mode  # 记录消融模式

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
                input_window_samples=input_window_samples,
                subject_id=subject_id  # <--- 新增：将被试ID传递给loader
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

        # 如果是 baseline_concat，直接拼接底层特征，不需要公共/私有编码器
        if self.ablation_mode == 'baseline_concat':
            fusion_dim = self.eeg_feature_net.out_dim + self.image_feature_net.out_dim
            print("🚀 [Ablation] 启动 Baseline 模式: 直接拼接特征，跳过解耦网络")
        else:
            fusion_dim = common_dim * 2 + private_dim * 2
            # 仅在非 baseline 时初始化解耦编码器
            self.common_encoder = nn.Sequential(
                nn.Linear(self.eeg_feature_net.out_dim, common_dim),
                nn.ReLU(), nn.Dropout(dropout_rate),
                nn.Linear(common_dim, common_dim), nn.ReLU()
            )
            self.eeg_private_encoder = nn.Sequential(
                nn.Linear(self.eeg_feature_net.out_dim, private_dim),
                nn.ReLU(), nn.Dropout(dropout_rate),
                nn.Linear(private_dim, private_dim), nn.ReLU()
            )
            self.image_private_encoder = nn.Sequential(
                nn.Linear(self.image_feature_net.out_dim, private_dim),
                nn.ReLU(), nn.Dropout(dropout_rate),
                nn.Linear(private_dim, private_dim), nn.ReLU()
            )

            if self.ablation_mode == 'no_cmd':
                print("🚀 [Ablation] 启动 No-CMD 模式: 去除公共特征对齐损失")
            elif self.ablation_mode == 'no_ortho':
                print("🚀 [Ablation] 启动 No-Ortho 模式: 去除私有特征正交损失")

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(128, n_classes if n_classes > 2 else 1)
        )

        # 初始化权重
        self._initialize_weights()

    def _build_eeg_path(self, eeg_model_name, in_chans, n_classes, input_window_samples, pretrained_ssbcinet):
        """构建EEG特征提取通路"""

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
            print("✅ 使用预训练的SSBCINet初始化EEG通路")
            feature_extractor = pretrained_ssbcinet.feature_extractor
            fusion_net = pretrained_ssbcinet.fusion
            out_dim = 512
            print("  ✅ 成功加载了脑电通路初始化权重")
        else:
            print("🔄 随机初始化EEG通路")
            feature_extractor = EEGFeatureExtractor(
                model_name=eeg_model_name,
                in_chans=in_chans,
                n_classes=n_classes,
                input_window_samples=input_window_samples,
            )
            fusion_net = EEGFusionNetwork(feature_extractor.out_dim)
            out_dim = 512

        return EEGFeaturePath(feature_extractor, fusion_net, out_dim)

    def _build_image_path(self, image_model_name, pretrained_image_model, image_model_type):
        """构建图像特征提取通路"""

        if pretrained_image_model is not None:
            print(f"✅ 使用预训练的{image_model_type.upper()}初始化图像通路")
            return ImageFeatureExtractor(
                base_model_name=image_model_name,
                pretrained_rsscnn=pretrained_image_model
            )
        else:
            print("🔄 随机初始化图像通路")
            return ImageFeatureExtractor(
                base_model_name=image_model_name,
                pretrained_rsscnn=None
            )

    def forward(self, eeg1, eeg2, img1, img2):
        """
        前向传播
        Returns:
            logits: 分类输出
            eeg_common: EEG公共特征
            image_common: 图像公共特征
            eeg_private: EEG私有特征
            image_private: 图像私有特征
        """
        # 提取基础特征
        eeg_base_features = self.eeg_feature_net(eeg1, eeg2)
        image_base_features = self.image_feature_net(img1, img2)

        if self.ablation_mode == 'baseline_concat':
            # 消融实验1：直接拼接
            fused_features = torch.cat([eeg_base_features, image_base_features], dim=1)
            eeg_common = image_common = eeg_private = image_private = None
        else:
            # 公共通道特征
            eeg_common = self.common_encoder(eeg_base_features)
            image_common = self.common_encoder(image_base_features)

            # 私有通道特征
            eeg_private = self.eeg_private_encoder(eeg_base_features)
            image_private = self.image_private_encoder(image_base_features)

            # 特征融合: EEG公共特征 + 图像公共特征 + EEG私有特征 + 图像私有特征
            fused_features = torch.cat([
                eeg_common,  # EEG公共特征
                image_common,  # 图像公共特征
                eeg_private,  # EEG私有特征
                image_private  # 图像私有特征
            ], dim=1)

        # 分类
        logits = self.classifier(fused_features)

        if logits.shape[1] == 1:
            logits = logits.squeeze()  # 二分类

        return logits, eeg_common, image_common, eeg_private, image_private

    def compute_loss(self, eeg_common, image_common, eeg_private, image_private, logits, targets):
        """
        计算总损失，包含三个部分：
        1. 分类损失
        2. 公共通道相似性损失
        3. 私有通道差异性损失
        """
        # 1. 分类损失
        if logits.dim() == 1:  # 二分类
            task_loss = F.binary_cross_entropy_with_logits(logits, targets.float())
        else:  # 多分类
            task_loss = F.cross_entropy(logits, targets)

        device = logits.device
        common_sim_loss = torch.tensor(0.0, device=device)
        private_diff_loss = torch.tensor(0.0, device=device)

        if self.ablation_mode == 'baseline_concat':
            total_loss = task_loss
        else:
            # 2. 公共通道相似性损失 (消融2：如果 no_cmd 则保持为0)
            if self.ablation_mode != 'no_cmd':
                common_sim_loss = self.cmd_loss(eeg_common, image_common, K=3)

            # 3. 私有通道差异性损失 (消融3：如果 no_ortho 则保持为0)
            if self.ablation_mode != 'no_ortho':
                private_diff_loss = self.orthogonality_loss(eeg_common, eeg_private, image_common, image_private)

            total_loss = task_loss + self.alpha * common_sim_loss + self.beta * private_diff_loss

        return {
            'total_loss': total_loss,
            'task_loss': task_loss,
            'common_sim_loss': common_sim_loss,
            'private_diff_loss': private_diff_loss
        }

    def cmd_loss(self, X, Y, K=3):
        """
        中心矩差异损失 (Central Moment Discrepancy)
        论文中使用的距离度量方法
        """
        # 一阶矩 (均值)
        x_mean = torch.mean(X, 0)
        y_mean = torch.mean(Y, 0)

        moment_diff = torch.norm(x_mean - y_mean, p=2)

        diffs = [moment_diff]  # 累积所有 moment 差异

        for k in range(2, K + 1):
            x_moment = torch.mean((X - x_mean) ** k, dim=0)
            y_moment = torch.mean((Y - y_mean) ** k, dim=0)
            diffs.append(torch.norm(x_moment - y_moment, p=2))

        return sum(diffs)

    def orthogonality_loss(self, eeg_common, eeg_private, image_common, image_private):
        """
        简单的正交性损失 - 直接处理维度不匹配
        """

        def dimension_aware_loss(A, B):
            """维度感知的损失计算"""
            # 统一到最小维度
            min_dim = min(A.size(1), B.size(1))
            A_trim = A[:, :min_dim]
            B_trim = B[:, :min_dim]

            # 归一化
            A_norm = F.normalize(A_trim, p=2, dim=1)
            B_norm = F.normalize(B_trim, p=2, dim=1)

            # 计算批次内余弦相似度的平均值
            cosine_sim = (A_norm * B_norm).sum(dim=1)  # [batch_size]

            # 我们希望余弦相似度接近0（正交）
            return cosine_sim.abs().mean()  # 取绝对值后平均

        loss1 = dimension_aware_loss(eeg_common, eeg_private)
        loss2 = dimension_aware_loss(image_common, image_private)
        loss3 = dimension_aware_loss(eeg_private, image_private)

        total_loss = (loss1 + loss2 + loss3) / 3.0

        return total_loss

    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)


# 测试修改后的网络
if __name__ == "__main__":
    print("=== 测试多模态网络 ===")


    model = MultiModalFusionNetwork(
        use_pretrained_eeg=True,  # 测试时设为False避免加载实际文件
        use_pretrained_image=True,
    )
    print(model)

    # 测试输入
    eeg1 = torch.randn(2, 64, 2000)
    eeg2 = torch.randn(2, 64, 2000)
    img1 = torch.randn(2, 3, 224, 224)
    img2 = torch.randn(2, 3, 224, 224)

    # 前向传播
    with torch.no_grad():
        logits, eeg_common, image_common, eeg_private, image_private = model(eeg1, eeg2, img1, img2)

    print(f"输出logits形状: {logits.shape}")