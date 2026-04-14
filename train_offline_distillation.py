import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import time
import glob
import json
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset

warnings.filterwarnings('ignore')

from models.multi_model import MultiModalFusionNetwork
from models.privileged_model import PrivilegedMultimodalNetwork
from MyPP2Dataset import MyPP2Dataset, create_dataloaders
from config.config_privileged_model import get_config


class OfflineDistillationTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 初始化离线蒸馏训练器 | 使用设备: {self.device}")

        # 1. 动态生成离线特权学习输出目录
        if getattr(config, 'is_path_locked', False):
            self.output_dir = config.output_dir
        else:
            self.output_dir = config.get_output_dir().replace('outputs_privileged', 'outputs_offline_distill')
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📂 输出目录: {self.output_dir}")

        # 2. 挂载教师网络 (双模态网络)
        self.teacher = self._load_teacher_network()

        # 3. 挂载学生网络 (特权网络)
        self.student = self._create_student_network()

        # 初始化历史记录字典
        self.history = {
            'train_losses': [], 'val_losses': [],
            'train_student_losses': [], 'val_student_losses': [],
            'train_distill_losses': [], 'val_distill_losses': [],
            'train_align_losses': [], 'val_align_losses': [],
            'train_student_acc': [], 'val_student_acc': [],
            'train_student_f1': [], 'val_student_f1': [],
            'train_student_auc': [], 'val_student_auc': [],
            'learning_rates': []
        }

        self.best_val_loss = float('inf')
        self.best_student_acc = 0.0

    def _load_teacher_network(self):
        """加载预训练双模态教师模型"""
        # 优先使用批量脚本传递进来的精确路径
        if hasattr(self.config, 'teacher_ckpt_path'):
            ckpt_path = self.config.teacher_ckpt_path
        else:
            raise ValueError("未指定教师模型路径 config.teacher_ckpt_path")

        print(f"🏫 正在加载教师模型权重: {ckpt_path}")

        teacher = MultiModalFusionNetwork(
            eeg_model_name=self.config.eeg_model_name,
            image_model_name=self.config.image_model_name,
            image_model_type=self.config.image_model_type,
            subject_id=self.config.subject_id,
            in_chans=self.config.in_chans,
            n_classes=self.config.n_classes,
            input_window_samples=self.config.input_window_samples,
            feature_dim=self.config.feature_dim,
            use_pretrained_eeg=False,
            use_pretrained_image=False
        ).to(self.device)

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        teacher.load_state_dict(checkpoint['model_state_dict'])

        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

        return teacher

    def _create_student_network(self):
        """加载学生网络并执行显存优化"""
        full_model = PrivilegedMultimodalNetwork(
            student_modality=self.config.student_modality,
            eeg_model_name=self.config.eeg_model_name,
            image_model_name=self.config.image_model_name,
            image_model_type=self.config.image_model_type,
            subject_id=self.config.subject_id,
            in_chans=self.config.in_chans,
            n_classes=self.config.n_classes,
            input_window_samples=self.config.input_window_samples,
            use_pretrained_eeg=self.config.use_pretrained,
            use_pretrained_image=self.config.use_pretrained,
            base_path=self.config.base_path,
            feature_dim=self.config.feature_dim,
            dropout_rate=self.config.dropout_rate,
            temperature=self.config.temperature
        ).to(self.device)

        # 显存优化核心操作：离线蒸馏不需要在线的 teacher_network，直接删除节省显存！
        if hasattr(full_model, 'teacher_network'):
            del full_model.teacher_network
            print("✂️ 已裁剪冗余的在线教师计算图，释放显存")

        return full_model

    def create_dataloaders(self):
        """支持 Intra-subject 和 Cross-subject (LOSO) 两种数据加载模式"""
        mode = getattr(self.config, 'train_mode', 'Intra-subject')

        if mode == "Intra-subject":
            print(f"📦 数据模式: Intra-subject | 划分被试 {self.config.subject_id} 的 80%训练 20%测试")
            dataset = MyPP2Dataset(is_flipped=False, transform=self.config.transform, subject_id=self.config.subject_id)
            train_loader, val_loader = create_dataloaders(dataset=dataset, batch_size=self.config.batch_size,
                                                          shuffle=True)

            print(f"  -> 训练集 Batch 数量: {len(train_loader)} | 测试集 Batch 数量: {len(val_loader)}")
            return train_loader, val_loader

        elif mode == "Cross-subject":
            print(f"🌍 数据模式: Cross-subject (LOSO) | 测试集为被试: {self.config.subject_id}")

            if not hasattr(self.config, 'all_subjects'):
                raise ValueError("Cross-subject 模式下需要在 config 中提供 all_subjects 列表")

            train_datasets = []
            # 遍历所有被试，排除当前测试被试，其余全部作为训练集
            for subj in self.config.all_subjects:
                if subj != self.config.subject_id:
                    ds = MyPP2Dataset(is_flipped=False, transform=self.config.transform, subject_id=subj)
                    train_datasets.append(ds)

            # 拼接 N-1 个被试的数据集
            train_dataset = ConcatDataset(train_datasets)
            # 当前被试作为完整的验证/测试集
            val_dataset = MyPP2Dataset(is_flipped=False, transform=self.config.transform,
                                       subject_id=self.config.subject_id)

            # 构建 DataLoader
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                drop_last=True  # 训练时丢弃不足 batch_size 的尾巴，保持维度稳定
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False
            )

            print(f"  -> 📦 融合训练集大小: {len(train_dataset)} 样本 (来自其他 {len(train_datasets)} 个被试)")
            print(f"  -> 📦 独立测试集大小: {len(val_dataset)} 样本 (仅来自被试 {self.config.subject_id})")

            return train_loader, val_loader
        else:
            raise ValueError(f"未知的训练模式: {mode}")

    # ==================== 冻结与解冻控制 ====================
    def freeze_student_backbone(self):
        """冻结学生网络的主干"""
        for p in self.student.student_network['feature_path'].parameters():
            p.requires_grad = False
        print("🔒 冻结学生网络 Backbone")

    def unfreeze_student_backbone_last_layers(self, layers=4):
        """解冻学生主干最后几层"""
        path_module = self.student.student_network['feature_path']
        target_module = None

        if self.config.student_modality == 'eeg':
            if hasattr(path_module, 'feature_extractor'):
                target_module = path_module.feature_extractor.feature_net
            if hasattr(path_module, 'fusion'):
                for p in path_module.fusion.parameters(): p.requires_grad = True
        else:
            if hasattr(path_module, 'feature_extractor'):
                target_module = path_module.feature_extractor.features
            elif hasattr(path_module, 'features'):
                target_module = path_module.features
            if hasattr(path_module, 'fusion_convs'):
                for p in path_module.fusion_convs.parameters(): p.requires_grad = True

        if target_module is not None:
            modules = list(target_module.children())
            if len(modules) > 0:
                for idx in list(range(len(modules)))[-layers:]:
                    for p in modules[idx].parameters(): p.requires_grad = True

        print(f"🔓 解冻学生网络 Backbone 最后 {layers} 层")

    def build_optimizer(self, lr_head, lr_backbone):
        head_params, backbone_params = [], []
        for name, p in self.student.student_network.named_parameters():
            if not p.requires_grad: continue
            if 'feature_path' in name:
                backbone_params.append(p)
            else:
                head_params.append(p)

        param_groups = []
        if head_params: param_groups.append({'params': head_params, 'lr': lr_head})
        if backbone_params: param_groups.append({'params': backbone_params, 'lr': lr_backbone})

        return optim.AdamW(param_groups, weight_decay=self.config.weight_decay)

    # ==================== 离线损失计算核心 ====================
    def compute_offline_loss(self, s_logits, t_logits, s_features, target_features, labels):
        # 1. 纯净的任务损失 (Task Loss)
        if s_logits.dim() == 1:
            s_loss = F.binary_cross_entropy_with_logits(s_logits, labels.float())
        else:
            s_loss = F.cross_entropy(s_logits, labels)

        # 2. 软标签蒸馏损失 (Distillation Loss)
        distill_loss = self.student.distillation_loss(t_logits, s_logits)

        # 3. 特征对齐损失 (Feature Alignment Loss)
        align_loss = F.mse_loss(s_features, target_features.detach())

        # 总损失 = 任务损失 + Gamma * 蒸馏 + 0.1 * 对齐
        total_loss = s_loss + self.config.gamma * distill_loss + 0.1 * align_loss

        return {
            'total_loss': total_loss,
            'student_loss': s_loss,
            'distill_loss': distill_loss,
            'align_loss': align_loss
        }

    # ==================== 训练与验证流 ====================
    def _run_epoch(self, dataloader, optimizer=None, mode='train'):
        is_train = mode == 'train'
        if is_train:
            self.student.train()
        else:
            self.student.eval()

        metrics_accum = {k: 0.0 for k in ['total_loss', 'student_loss', 'distill_loss', 'align_loss']}
        all_labels, all_preds, all_probs = [], [], []

        torch.set_grad_enabled(is_train)

        for left_img, right_img, left_eeg, right_eeg, labels in dataloader:
            left_img, right_img = left_img.to(self.device), right_img.to(self.device)
            left_eeg, right_eeg = left_eeg.to(self.device), right_eeg.to(self.device)
            labels = labels.to(self.device)

            if is_train: optimizer.zero_grad()

            # --- 步骤 1: 教师离线推断 ---
            with torch.no_grad():
                t_logits, t_eeg_com, t_img_com, _, _ = self.teacher(left_eeg, right_eeg, left_img, right_img)

            # --- 步骤 2: 学生推断 ---
            if self.config.student_modality == 'image':
                base_feat = self.student.student_network['feature_path'](left_img, right_img)
                target_feat = t_img_com
            else:
                base_feat = self.student.student_network['feature_path'](left_eeg, right_eeg)
                target_feat = t_eeg_com

            s_feat = self.student.student_network['feature_encoder'](base_feat)
            s_logits = self.student.student_network['classifier'](s_feat)
            if s_logits.shape[1] == 1: s_logits = s_logits.squeeze()

            # --- 步骤 3: 损失计算 ---
            loss_dict = self.compute_offline_loss(s_logits, t_logits, s_feat, target_feat, labels)

            if is_train:
                loss_dict['total_loss'].backward()
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.config.grad_clip)
                optimizer.step()

            # 统计
            for k in metrics_accum.keys():
                metrics_accum[k] += loss_dict[k].item()

            if s_logits.dim() == 1:
                probs = torch.sigmoid(s_logits)
                preds = (probs > 0.5).long()
            else:
                probs = F.softmax(s_logits, dim=1)[:, 1]
                preds = torch.argmax(s_logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

        # 汇总指标
        from sklearn.metrics import accuracy_score
        num_batches = len(dataloader)
        metrics = {k: v / num_batches for k, v in metrics_accum.items()}
        metrics['acc'] = accuracy_score(all_labels, all_preds) * 100
        metrics['f1'] = f1_score(all_labels, all_preds, average='macro')
        try:
            metrics['auc'] = roc_auc_score(all_labels, all_probs)
        except ValueError:
            metrics['auc'] = 0.5

        return metrics

    def train_stage(self, stage_name, train_loader, val_loader, lr_head, lr_backbone, max_epochs, patience, min_epochs):
        print(f"\n{'=' * 50}\n🏁 启动离线阶段: {stage_name}\n{'=' * 50}")
        optimizer = self.build_optimizer(lr_head, lr_backbone)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=min_epochs, T_mult=1,
                                                                   eta_min=lr_head * 0.01)

        stage_best_loss = float('inf')
        no_improve = 0

        for epoch in range(max_epochs):
            train_metrics = self._run_epoch(train_loader, optimizer, mode='train')
            val_metrics = self._run_epoch(val_loader, mode='val')
            scheduler.step()

            # 记录历史
            self.history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            for prefix, metrics in [('train', train_metrics), ('val', val_metrics)]:
                self.history[f'{prefix}_losses'].append(metrics['total_loss'])
                self.history[f'{prefix}_student_losses'].append(metrics['student_loss'])
                self.history[f'{prefix}_distill_losses'].append(metrics['distill_loss'])
                self.history[f'{prefix}_align_losses'].append(metrics['align_loss'])
                self.history[f'{prefix}_student_acc'].append(metrics['acc'])
                self.history[f'{prefix}_student_f1'].append(metrics['f1'])
                self.history[f'{prefix}_student_auc'].append(metrics['auc'])

            # 评估与保存
            if val_metrics['total_loss'] < stage_best_loss:
                stage_best_loss = val_metrics['total_loss']
                no_improve = 0
                torch.save({'state_dict': self.student.student_network.state_dict()},
                           os.path.join(self.output_dir, f"best_{stage_name}.pth"))
            else:
                no_improve += 1

            # 全局最优
            if val_metrics['total_loss'] < self.best_val_loss: self.best_val_loss = val_metrics['total_loss']
            if val_metrics['acc'] > self.best_student_acc: self.best_student_acc = val_metrics['acc']

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"[{stage_name}] Epoch {epoch + 1}/{max_epochs} | Lr: {optimizer.param_groups[0]['lr']:.2e} | 停滞: {no_improve}/{patience}")
                print(f"  损失 -> Train: {train_metrics['total_loss']:.4f} | Val: {val_metrics['total_loss']:.4f}")
                print(f"  Acc  -> Train: {train_metrics['acc']:.2f}% | Val: {val_metrics['acc']:.2f}%")

            if no_improve >= patience and epoch >= min_epochs:
                print(f"🛑 触发早停: {stage_name}")
                break

    def train(self):
        train_loader, val_loader = self.create_dataloaders()

        # Phase 1: 冻结骨干，仅训练头部对齐
        self.freeze_student_backbone()
        self.train_stage("stage1_freeze_backbone", train_loader, val_loader, lr_head=1e-3, lr_backbone=0, max_epochs=30,
                         patience=8, min_epochs=10)

        # Phase 2: 解冻骨干末端，全面收敛
        layers = 6 if self.config.image_model_name == 'VGG' else 4
        self.unfreeze_student_backbone_last_layers(layers=layers)
        self.train_stage("stage2_finetune_backbone", train_loader, val_loader, lr_head=3e-4, lr_backbone=2e-5,
                         max_epochs=40, patience=10, min_epochs=15)

        print(
            f"\n🎉 离线蒸馏全部完成！\n🏆 最佳 Val Loss: {self.best_val_loss:.4f} | 最佳 Val Acc: {self.best_student_acc:.2f}%")
        torch.save(self.history, os.path.join(self.output_dir, 'offline_history.pth'))
        self._plot_results()

    def _plot_results(self):
        sns.set_theme(style="whitegrid", context="paper")
        epochs = range(1, len(self.history['train_losses']) + 1)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        # 1. Losses
        axes[0].plot(epochs, self.history['train_losses'], label='Train Total Loss')
        axes[0].plot(epochs, self.history['val_losses'], label='Val Total Loss')
        axes[0].plot(epochs, self.history['train_distill_losses'], label='Train Distill (x Gamma)', linestyle='--')
        axes[0].set_title('Loss Progression', fontweight='bold')
        axes[0].legend()

        # 2. Accuracy
        axes[1].plot(epochs, self.history['train_student_acc'], label='Train Acc')
        axes[1].plot(epochs, self.history['val_student_acc'], label='Val Acc')
        axes[1].set_title('Student Accuracy', fontweight='bold')
        axes[1].legend()

        # 3. AUC/F1
        axes[2].plot(epochs, self.history['val_student_f1'], label='Val F1')
        axes[2].plot(epochs, self.history['val_student_auc'], label='Val AUC')
        axes[2].set_title('F1 & AUC Metrics', fontweight='bold')
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'offline_distill_curves.png'), dpi=300)
        plt.close()
        sns.reset_orig()


if __name__ == "__main__":
    config = get_config('default')
    trainer = OfflineDistillationTrainer(config)
    trainer.train()