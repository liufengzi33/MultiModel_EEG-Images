import torch
import torch.nn as nn
import torch.nn.functional as F
from models.eeg_models import EEGFeatureExtractor, EEGFusionNetwork
from utils.model_loader import ModelLoader
from models.image_models import ImageFeatureExtractor


class PrivilegedMultimodalNetwork(nn.Module):
    def __init__(self,
                 eeg_model_name="EEGNetv1",
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
                 gamma=0.3,  # 知识蒸馏损失权重
                 temperature=2.0):  # 蒸馏温度
        """
        特权学习多模态网络

        Args:
            gamma: 知识蒸馏损失权重
            temperature: 蒸馏温度参数
        """
        super(PrivilegedMultimodalNetwork, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
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

        # 1. 构建完整的多模态教师网络（训练时使用）
        self.teacher_network = self._build_teacher_network(
            eeg_model_name=eeg_model_name,
            image_model_name=image_model_name,
            image_model_type=image_model_type,
            in_chans=in_chans,
            n_classes=n_classes,
            input_window_samples=input_window_samples,
            pretrained_ssbcinet=pretrained_ssbcinet,
            pretrained_image_model=pretrained_image_model,
            common_dim=common_dim,
            private_dim=private_dim,
            dropout_rate=dropout_rate
        )

        # 2. 构建仅图像的学生网络（测试时使用）
        self.student_network = self._build_student_network(
            image_model_name=image_model_name,
            pretrained_image_model=pretrained_image_model,
            image_model_type=image_model_type,
            common_dim=common_dim,
            private_dim=private_dim,
            dropout_rate=dropout_rate,
            n_classes=n_classes
        )

    def _build_teacher_network(self, eeg_model_name, image_model_name, image_model_type,
                               in_chans, n_classes, input_window_samples,
                               pretrained_ssbcinet, pretrained_image_model,
                               common_dim, private_dim, dropout_rate):
        """构建完整的教师网络（多模态）"""

        # EEG特征提取通路
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
            print("✅ 使用预训练的SSBCINet初始化教师网络EEG通路")
            eeg_feature_extractor = pretrained_ssbcinet.feature_extractor
            eeg_fusion_net = pretrained_ssbcinet.fusion
            eeg_out_dim = 512
        else:
            print("🔄 随机初始化教师网络EEG通路")
            eeg_feature_extractor = EEGFeatureExtractor(
                model_name=eeg_model_name,
                in_chans=in_chans,
                n_classes=n_classes,
                input_window_samples=input_window_samples,
            )
            eeg_fusion_net = EEGFusionNetwork(eeg_feature_extractor.out_dim)
            eeg_out_dim = 512

        eeg_path = EEGFeaturePath(eeg_feature_extractor, eeg_fusion_net, eeg_out_dim)

        # 图像特征提取通路
        if pretrained_image_model is not None:
            print(f"✅ 使用预训练的{image_model_type.upper()}初始化教师网络图像通路")
            image_path = ImageFeatureExtractor(
                base_model_name=image_model_name,
                pretrained_rsscnn=pretrained_image_model
            )
        else:
            print("🔄 随机初始化教师网络图像通路")
            image_path = ImageFeatureExtractor(
                base_model_name=image_model_name,
                pretrained_rsscnn=None
            )

        # 教师网络的特征编码器和分类器
        common_encoder = nn.Sequential(
            nn.Linear(eeg_out_dim, common_dim),  # 注意：这里假设EEG和图像特征维度相同
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(common_dim, common_dim),
            nn.ReLU()
        )

        eeg_private_encoder = nn.Sequential(
            nn.Linear(eeg_out_dim, private_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(private_dim, private_dim),
            nn.ReLU()
        )

        image_private_encoder = nn.Sequential(
            nn.Linear(image_path.out_dim, private_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(private_dim, private_dim),
            nn.ReLU()
        )

        fusion_dim = common_dim + private_dim * 2
        classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(128, n_classes if n_classes > 2 else 1)
        )

        return nn.ModuleDict({
            'eeg_path': eeg_path,
            'image_path': image_path,
            'common_encoder': common_encoder,
            'eeg_private_encoder': eeg_private_encoder,
            'image_private_encoder': image_private_encoder,
            'classifier': classifier
        })

    def _build_student_network(self, image_model_name, pretrained_image_model,
                               image_model_type, common_dim, private_dim,
                               dropout_rate, n_classes):
        """构建仅图像的学生网络"""

        # 图像特征提取
        if pretrained_image_model is not None:
            print(f"✅ 使用预训练的{image_model_type.upper()}初始化学生网络")
            image_path = ImageFeatureExtractor(
                base_model_name=image_model_name,
                pretrained_rsscnn=pretrained_image_model
            )
        else:
            print("🔄 随机初始化学生网络")
            image_path = ImageFeatureExtractor(
                base_model_name=image_model_name,
                pretrained_rsscnn=None
            )

        # 学生网络的特征编码器和分类器
        feature_encoder = nn.Sequential(
            nn.Linear(image_path.out_dim, common_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(common_dim, common_dim),
            nn.ReLU()
        )

        classifier = nn.Sequential(
            nn.Linear(common_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(128, n_classes if n_classes > 2 else 1)
        )

        return nn.ModuleDict({
            'image_path': image_path,
            'feature_encoder': feature_encoder,
            'classifier': classifier
        })

    def forward(self, eeg1=None, eeg2=None, img1=None, img2=None, mode='train'):
        """
        前向传播

        Args:
            mode: 'train' - 训练模式，使用完整多模态信息
                  'test' - 测试模式，仅使用图像信息
        """
        if mode == 'train':
            return self._forward_train(eeg1, eeg2, img1, img2)
        else:
            return self._forward_test(img1, img2)

    def _forward_train(self, eeg1, eeg2, img1, img2):
        """训练阶段前向传播 - 使用完整多模态信息"""

        # 教师网络（多模态）前向传播
        eeg_base_features = self.teacher_network['eeg_path'](eeg1, eeg2)
        image_base_features_teacher = self.teacher_network['image_path'](img1, img2)

        # 公共通道特征
        eeg_common = self.teacher_network['common_encoder'](eeg_base_features)
        image_common_teacher = self.teacher_network['common_encoder'](image_base_features_teacher)

        # 私有通道特征
        eeg_private = self.teacher_network['eeg_private_encoder'](eeg_base_features)
        image_private_teacher = self.teacher_network['image_private_encoder'](image_base_features_teacher)

        # 特征融合
        fused_features = torch.cat([eeg_common, eeg_private, image_private_teacher], dim=1)

        # 教师网络分类输出
        teacher_logits = self.teacher_network['classifier'](fused_features)
        if teacher_logits.shape[1] == 1:
            teacher_logits = teacher_logits.squeeze()

        # 学生网络（仅图像）前向传播
        image_base_features_student = self.student_network['image_path'](img1, img2)
        student_features = self.student_network['feature_encoder'](image_base_features_student)
        student_logits = self.student_network['classifier'](student_features)
        if student_logits.shape[1] == 1:
            student_logits = student_logits.squeeze()

        return {
            'teacher_logits': teacher_logits,
            'student_logits': student_logits,
            'eeg_common': eeg_common,
            'image_common_teacher': image_common_teacher,
            'eeg_private': eeg_private,
            'image_private_teacher': image_private_teacher,
            'student_features': student_features
        }

    def _forward_test(self, img1, img2):
        """测试阶段前向传播 - 仅使用图像信息"""
        image_base_features = self.student_network['image_path'](img1, img2)
        features = self.student_network['feature_encoder'](image_base_features)
        logits = self.student_network['classifier'](features)

        if logits.shape[1] == 1:
            logits = logits.squeeze()

        return logits

    def compute_loss(self, outputs, targets):
        """
        计算特权学习总损失

        包含：
        1. 教师网络分类损失
        2. 学生网络分类损失
        3. 知识蒸馏损失
        4. 特征对齐损失
        """
        teacher_logits = outputs['teacher_logits']
        student_logits = outputs['student_logits']
        eeg_common = outputs['eeg_common']
        image_common_teacher = outputs['image_common_teacher']
        eeg_private = outputs['eeg_private']
        image_private_teacher = outputs['image_private_teacher']
        student_features = outputs['student_features']

        # 1. 教师网络分类损失
        if teacher_logits.dim() == 1:
            teacher_loss = F.binary_cross_entropy_with_logits(teacher_logits, targets.float())
        else:
            teacher_loss = F.cross_entropy(teacher_logits, targets)

        # 2. 学生网络分类损失
        if student_logits.dim() == 1:
            student_loss = F.binary_cross_entropy_with_logits(student_logits, targets.float())
        else:
            student_loss = F.cross_entropy(student_logits, targets)

        # 3. 知识蒸馏损失
        distill_loss = self.distillation_loss(teacher_logits, student_logits)

        # 4. 特征对齐损失（让学生网络学习教师网络的特征表示）
        feature_align_loss = F.mse_loss(
            student_features,
            image_common_teacher.detach()  # 使用教师网络的图像公共特征作为目标
        )

        # 5. 多模态一致性损失（原网络中的损失）
        common_sim_loss = self.cmd_loss(eeg_common, image_common_teacher, K=3)
        private_diff_loss = self.orthogonality_loss(eeg_common, eeg_private,
                                                    image_common_teacher, image_private_teacher)

        # 总损失
        total_loss = (teacher_loss + student_loss +
                      self.gamma * distill_loss +
                      0.1 * feature_align_loss +  # 特征对齐损失权重
                      self.alpha * common_sim_loss +
                      self.beta * private_diff_loss)

        return {
            'total_loss': total_loss,
            'teacher_loss': teacher_loss,
            'student_loss': student_loss,
            'distill_loss': distill_loss,
            'feature_align_loss': feature_align_loss,
            'common_sim_loss': common_sim_loss,
            'private_diff_loss': private_diff_loss
        }

    def distillation_loss(self, teacher_logits, student_logits):
        """知识蒸馏损失"""
        if teacher_logits.dim() == 1:  # 二分类
            # 将logits转换为概率
            teacher_probs = torch.sigmoid(teacher_logits / self.temperature)
            student_probs = torch.sigmoid(student_logits / self.temperature)
            distill_loss = F.binary_cross_entropy(student_probs, teacher_probs.detach())
        else:  # 多分类
            teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
            student_probs = F.softmax(student_logits / self.temperature, dim=1)
            distill_loss = F.kl_div(
                student_probs.log(),
                teacher_probs.detach(),
                reduction='batchmean'
            ) * (self.temperature ** 2)

        return distill_loss

    def cmd_loss(self, X, Y, K=3):
        """中心矩差异损失"""
        x_mean = torch.mean(X, 0)
        y_mean = torch.mean(Y, 0)
        moment_diff = torch.norm(x_mean - y_mean, p=2)
        diffs = [moment_diff]

        for k in range(2, K + 1):
            x_moment = torch.mean((X - x_mean) ** k, dim=0)
            y_moment = torch.mean((Y - y_mean) ** k, dim=0)
            diffs.append(torch.norm(x_moment - y_moment, p=2))

        return sum(diffs)

    def orthogonality_loss(self, eeg_common, eeg_private, image_common, image_private):
        """正交性损失"""

        def dimension_aware_loss(A, B):
            min_dim = min(A.size(1), B.size(1))
            A_trim = A[:, :min_dim]
            B_trim = B[:, :min_dim]
            A_norm = F.normalize(A_trim, p=2, dim=1)
            B_norm = F.normalize(B_trim, p=2, dim=1)
            cosine_sim = (A_norm * B_norm).sum(dim=1)
            return cosine_sim.abs().mean()

        loss1 = dimension_aware_loss(eeg_common, eeg_private)
        loss2 = dimension_aware_loss(image_common, image_private)
        loss3 = dimension_aware_loss(eeg_private, image_private)

        return (loss1 + loss2 + loss3) / 3.0

# 测试特权学习网络
if __name__ == "__main__":
    print("=== 测试特权学习网络 ===")

    # 创建模型
    model = PrivilegedMultimodalNetwork(
        use_pretrained_eeg=True,
        use_pretrained_image=True,
    )

    print("模型结构:")
    print(model)

    # 测试训练模式
    print("\n--- 训练模式测试 (多模态) ---")
    eeg1 = torch.randn(2, 64, 2000)
    eeg2 = torch.randn(2, 64, 2000)
    img1 = torch.randn(2, 3, 224, 224)
    img2 = torch.randn(2, 3, 224, 224)
    targets = torch.randint(0, 2, (2,)).float()

    # 训练前向传播
    outputs = model(eeg1, eeg2, img1, img2, mode='train')
    print(f"教师logits形状: {outputs['teacher_logits'].shape}")
    print(f"学生logits形状: {outputs['student_logits'].shape}")

    # 计算损失
    losses = model.compute_loss(outputs, targets)
    print(f"总损失: {losses['total_loss']:.4f}")
    print(f"教师损失: {losses['teacher_loss']:.4f}")
    print(f"学生损失: {losses['student_loss']:.4f}")
    print(f"蒸馏损失: {losses['distill_loss']:.4f}")

    # 测试测试模式
    print("\n--- 测试模式测试 (仅图像) ---")
    test_logits = model(img1=img1, img2=img2, mode='test')
    print(f"测试logits形状: {test_logits.shape}")