import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from utils.model_loader import ModelLoader


class PrivilegedLearningNetwork(nn.Module):
    def __init__(self,
                 eeg_model_name="EEGNet",
                 image_model_name="PlacesNet",
                 image_model_type="rsscnn",
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
                 beta=0.5,  # 私有损失权重
                 gamma=0.1):  # 蒸馏损失权重
        """
        特权学习网络 - 基于脑机耦合学习

        Args:
            gamma: 知识蒸馏损失权重，让学生网络模仿教师网络
        """
        super(PrivilegedLearningNetwork, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_classes = n_classes

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

        # 初始化特征提取通路
        self.eeg_feature_net = self._build_eeg_path(
            eeg_model_name=eeg_model_name,
            in_chans=in_chans,
            n_classes=n_classes,
            input_window_samples=input_window_samples,
            pretrained_ssbcinet=pretrained_ssbcinet
        )

        self.image_feature_net = self._build_image_path(
            image_model_name=image_model_name,
            pretrained_image_model=pretrained_image_model,
            image_model_type=image_model_type
        )

        # 公共通道编码器 (共享参数)
        self.common_encoder = nn.Sequential(
            nn.Linear(self.eeg_feature_net.out_dim, common_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(common_dim, common_dim),
            nn.ReLU()
        )

        # 私有通道编码器
        self.eeg_private_encoder = nn.Sequential(
            nn.Linear(self.eeg_feature_net.out_dim, private_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(private_dim, private_dim),
            nn.ReLU()
        )

        self.image_private_encoder = nn.Sequential(
            nn.Linear(self.image_feature_net.out_dim, private_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(private_dim, private_dim),
            nn.ReLU()
        )

        # 教师分类器 (使用EEG+图像，训练时用)
        teacher_fusion_dim = common_dim + private_dim * 2
        self.teacher_classifier = nn.Sequential(
            nn.Linear(teacher_fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(128, n_classes)
        )

        # 学生分类器 (仅使用图像，测试时用)
        student_fusion_dim = common_dim + private_dim  # 仅图像公共特征 + 图像私有特征
        self.student_classifier = nn.Sequential(
            nn.Linear(student_fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(128, n_classes if n_classes > 2 else 1)
        )

        # 初始化权重
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
            print("✅ 使用预训练的SSBCINet初始化EEG通路")
            feature_extractor = pretrained_ssbcinet.feature_extractor
            fusion_net = pretrained_ssbcinet.fusion
            out_dim = 512
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
        from image_models import ImageFeatureExtractor

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

    def forward(self, eeg1, eeg2, img1, img2, mode='train'):
        """
        前向传播

        Args:
            mode: 'train' - 训练模式，使用EEG和图像
                  'test' - 测试模式，仅使用图像
        """
        if mode == 'train':
            return self._forward_train(eeg1, eeg2, img1, img2)
        else:
            return self._forward_test(img1, img2)

    def _forward_train(self, eeg1, eeg2, img1, img2):
        """训练模式前向传播 - 使用EEG和图像"""
        # 提取基础特征
        eeg_base_features = self.eeg_feature_net(eeg1, eeg2)
        image_base_features = self.image_feature_net(img1, img2)

        # 公共通道特征
        eeg_common = self.common_encoder(eeg_base_features)
        image_common = self.common_encoder(image_base_features)

        # 私有通道特征
        eeg_private = self.eeg_private_encoder(eeg_base_features)
        image_private = self.image_private_encoder(image_base_features)

        # 教师网络融合特征 (EEG公共 + EEG私有 + 图像私有)
        teacher_fused = torch.cat([eeg_common, eeg_private, image_private], dim=1)
        teacher_logits = self.teacher_classifier(teacher_fused)

        # 学生网络融合特征 (图像公共 + 图像私有)
        student_fused = torch.cat([image_common, image_private], dim=1)
        student_logits = self.student_classifier(student_fused)

        return {
            'teacher_logits': teacher_logits,
            'student_logits': student_logits,
            'eeg_common': eeg_common,
            'image_common': image_common,
            'eeg_private': eeg_private,
            'image_private': image_private
        }

    def _forward_test(self, img1, img2):
        """测试模式前向传播 - 仅使用图像"""
        # 提取图像基础特征
        image_base_features = self.image_feature_net(img1, img2)

        # 图像公共通道特征
        image_common = self.common_encoder(image_base_features)

        # 图像私有通道特征
        image_private = self.image_private_encoder(image_base_features)

        # 学生网络融合特征 (图像公共 + 图像私有)
        student_fused = torch.cat([image_common, image_private], dim=1)
        student_logits = self.student_classifier(student_fused)

        if self.n_classes <= 2 and student_logits.shape[1] == 1:
            student_logits = student_logits.squeeze()

        return student_logits

    def compute_loss(self, outputs, targets, temperature=2.0):
        """
        计算特权学习的总损失

        Args:
            temperature: 知识蒸馏的温度参数
        """
        teacher_logits = outputs['teacher_logits']
        student_logits = outputs['student_logits']
        eeg_common = outputs['eeg_common']
        image_common = outputs['image_common']
        eeg_private = outputs['eeg_private']
        image_private = outputs['image_private']

        # 1. 教师网络分类损失 (使用特权信息EEG)
        teacher_loss = F.cross_entropy(teacher_logits, targets)

        # 2. 学生网络分类损失
        if self.n_classes <= 2:
            student_loss = F.binary_cross_entropy_with_logits(
                student_logits, targets.float()
            )
        else:
            student_loss = F.cross_entropy(student_logits, targets)

        # 3. 知识蒸馏损失 - 让学生模仿教师的输出分布
        distill_loss = self.knowledge_distillation_loss(
            teacher_logits, student_logits, temperature
        )

        # 4. 公共通道相似性损失
        common_sim_loss = self.cmd_loss(eeg_common, image_common, K=3)

        # 5. 私有通道差异性损失
        private_diff_loss = self.orthogonality_loss(
            eeg_common, eeg_private, image_common, image_private
        )

        # 总损失
        total_loss = (teacher_loss + student_loss +
                      self.gamma * distill_loss +
                      self.alpha * common_sim_loss +
                      self.beta * private_diff_loss)

        return {
            'total_loss': total_loss,
            'teacher_loss': teacher_loss,
            'student_loss': student_loss,
            'distill_loss': distill_loss,
            'common_sim_loss': common_sim_loss,
            'private_diff_loss': private_diff_loss
        }

    def knowledge_distillation_loss(self, teacher_logits, student_logits, temperature):
        """知识蒸馏损失 - 让学生网络模仿教师网络的输出分布"""
        # 使用softmax-temperature软化概率分布
        teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=1)

        # KL散度损失
        distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        distill_loss *= (temperature ** 2)  # 缩放损失

        return distill_loss

    def cmd_loss(self, X, Y, K=3):
        """中心矩差异损失"""
        x_mean = torch.mean(X, 0)
        y_mean = torch.mean(Y, 0)
        moment_diff = torch.norm(x_mean - y_mean, 2)

        for k in range(2, K + 1):
            x_moment = torch.mean((X - x_mean) ** k, 0)
            y_moment = torch.mean((Y - y_mean) ** k, 0)
            moment_diff += torch.norm(x_moment - y_moment, 2)

        return moment_diff

    def orthogonality_loss(self, eeg_common, eeg_private, image_common, image_private):
        """正交性损失"""
        batch_size = eeg_common.size(0)

        eeg_orth_loss = torch.norm(torch.mm(eeg_common.t(), eeg_private), p='fro') ** 2
        image_orth_loss = torch.norm(torch.mm(image_common.t(), image_private), p='fro') ** 2
        cross_orth_loss = torch.norm(torch.mm(eeg_private.t(), image_private), p='fro') ** 2

        return (eeg_orth_loss + image_orth_loss + cross_orth_loss) / batch_size

    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)


# 测试特权学习网络
if __name__ == "__main__":
    print("=== 测试特权学习网络 ===")

    model = PrivilegedLearningNetwork(
        use_pretrained_eeg=False,
        use_pretrained_image=False,
        n_classes=7  # 7类情感分类
    )
    print(model)
    # 训练模式测试
    print("\n=== 训练模式 ===")
    eeg1 = torch.randn(2, 64, 2000)
    eeg2 = torch.randn(2, 64, 2000)
    img1 = torch.randn(2, 3, 224, 224)
    img2 = torch.randn(2, 3, 224, 224)
    targets = torch.tensor([0, 1])  # 分类标签

    outputs = model(eeg1, eeg2, img1, img2, mode='train')
    losses = model.compute_loss(outputs, targets)

    print(f"教师网络输出形状: {outputs['teacher_logits'].shape}")
    print(f"学生网络输出形状: {outputs['student_logits'].shape}")

    for loss_name, loss_value in losses.items():
        print(f"{loss_name}: {loss_value.item():.4f}")

    # 测试模式测试
    print("\n=== 测试模式 ===")
    test_output = model(eeg1, eeg2, img1, img2, mode='test')
    print(f"测试模式输出形状: {test_output.shape}")

    print("\n✅ 特权学习网络测试完成！")
    print("训练时：使用EEG+图像，通过知识蒸馏让学生网络学习教师网络的知识")
    print("测试时：仅使用图像，学生网络独立完成分类任务")