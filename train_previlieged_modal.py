import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import warnings
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

warnings.filterwarnings('ignore')
sns.set_theme()

from models.privileged_model import PrivilegedMultimodalNetwork
from MyPP2Dataset import MyPP2Dataset, create_dataloaders
from config.config_privileged_model import get_config


class PrivilegedTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 创建动态输出目录
        self.output_dir = config.get_output_dir()
        self.config.output_dir = self.output_dir  # 更新config中的输出目录
        print(f"输出目录: {self.output_dir}")

        # 创建模型
        self.model = self._create_model()

        # 优化器和损失函数
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.train_teacher_losses = []
        self.train_student_losses = []
        self.train_distill_losses = []
        self.train_feature_align_losses = []
        self.train_common_sim_losses = []
        self.train_private_diff_losses = []
        self.val_teacher_losses = []
        self.val_student_losses = []
        self.val_distill_losses = []
        self.val_feature_align_losses = []
        self.val_common_sim_losses = []
        self.val_private_diff_losses = []

        # 准确率记录 - 新增训练准确率
        self.train_teacher_accuracies = []  # 新增
        self.train_student_accuracies = []  # 新增
        self.val_teacher_accuracies = []
        self.val_student_accuracies = []
        self.learning_rates = []

        self.best_val_loss = float('inf')
        self.best_teacher_acc = 0.0
        self.best_student_acc = 0.0
        self.current_epoch = 0

    def _create_model(self):
        """创建特权学习模型"""
        model = PrivilegedMultimodalNetwork(
            eeg_model_name=self.config.eeg_model_name,
            image_model_name=self.config.image_model_name,
            image_model_type=self.config.image_model_type,
            in_chans=self.config.in_chans,
            n_classes=self.config.n_classes,
            input_window_samples=self.config.input_window_samples,
            use_pretrained_eeg=self.config.use_pretrained,
            use_pretrained_image=self.config.use_pretrained,
            base_path=self.config.base_path,
            common_dim=self.config.common_dim,
            private_dim=self.config.private_dim,
            dropout_rate=self.config.dropout_rate,
            alpha=self.config.alpha,
            beta=self.config.beta,
            gamma=self.config.gamma,
            temperature=self.config.temperature
        )

        return model.to(self.device)

    def _create_optimizer(self):
        """创建优化器"""
        if self.config.optimizer.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        elif self.config.optimizer.lower() == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        elif self.config.optimizer.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.config.lr, momentum=self.config.momentum,
                             weight_decay=self.config.weight_decay)
        else:
            raise ValueError(f"不支持的优化器: {self.config.optimizer}")

    def _create_scheduler(self):
        """创建学习率调度器"""
        if self.config.scheduler.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs
            )
        elif self.config.scheduler.lower() == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.config.scheduler.lower() == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.config.patience,
                factor=self.config.factor,
                verbose=True
            )
        else:
            return None

    def create_dataloaders(self):
        """创建数据加载器"""
        print("创建数据加载器...")

        # 创建数据集
        dataset = MyPP2Dataset(
            is_flipped=False,
            transform=self.config.transform,
            subject_id=self.config.subject_id
        )
        # 创建数据加载器
        train_loader, val_loader = create_dataloaders(
            dataset=dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        # 检查数据分布
        train_labels = [label for _, _, _, _, label in train_loader.dataset]
        val_labels = [label for _, _, _, _, label in val_loader.dataset]

        print(f"训练集分布: {torch.bincount(torch.tensor(train_labels)).tolist()}")
        print(f"验证集分布: {torch.bincount(torch.tensor(val_labels)).tolist()}")
        print(f"训练集: {len(train_loader.dataset)}, 验证集: {len(val_loader.dataset)}")

        return train_loader, val_loader

    def _calculate_accuracy(self, logits, labels):
        """计算准确率"""
        if logits.dim() == 1:  # 二分类
            preds = (torch.sigmoid(logits) > 0.5).float()
        else:  # 多分类
            preds = torch.argmax(logits, dim=1)
        return (preds == labels).float().mean().item() * 100

    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_teacher_loss = 0
        total_student_loss = 0
        total_distill_loss = 0
        total_feature_align_loss = 0
        total_common_sim_loss = 0
        total_private_diff_loss = 0

        # 新增：训练准确率计算
        teacher_correct = 0
        student_correct = 0
        total_samples = 0

        for batch_idx, (left_img, right_img, left_eeg, right_eeg, labels) in enumerate(train_loader):
            # 移动到设备
            left_img = left_img.to(self.device)
            right_img = right_img.to(self.device)
            left_eeg = left_eeg.to(self.device)
            right_eeg = right_eeg.to(self.device)
            labels = labels.to(self.device)

            # 清零梯度
            self.optimizer.zero_grad()

            # 前向传播
            outputs = self.model(
                eeg1=left_eeg, eeg2=right_eeg,
                img1=left_img, img2=right_img,
                mode='train'
            )

            # 计算损失
            loss_dict = self.model.compute_loss(outputs, labels)
            loss = loss_dict['total_loss']

            # 反向传播
            loss.backward()

            # 梯度裁剪
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip
                )

            # 更新参数
            self.optimizer.step()

            # 记录所有损失
            total_loss += loss.item()
            total_teacher_loss += loss_dict['teacher_loss'].item()
            total_student_loss += loss_dict['student_loss'].item()
            total_distill_loss += loss_dict['distill_loss'].item()
            total_feature_align_loss += loss_dict['feature_align_loss'].item()
            total_common_sim_loss += loss_dict['common_sim_loss'].item()
            total_private_diff_loss += loss_dict['private_diff_loss'].item()

            # 新增：计算训练准确率
            teacher_logits = outputs['teacher_logits']
            student_logits = outputs['student_logits']

            if teacher_logits.dim() == 1:  # 二分类
                teacher_preds = (torch.sigmoid(teacher_logits) > 0.5).float()
                student_preds = (torch.sigmoid(student_logits) > 0.5).float()
            else:  # 多分类
                teacher_preds = torch.argmax(teacher_logits, dim=1)
                student_preds = torch.argmax(student_logits, dim=1)

            teacher_correct += (teacher_preds == labels).sum().item()
            student_correct += (student_preds == labels).sum().item()
            total_samples += labels.size(0)

        # 计算平均损失和准确率
        num_batches = len(train_loader)
        teacher_acc = teacher_correct / total_samples * 100
        student_acc = student_correct / total_samples * 100

        return {
            'total_loss': total_loss / num_batches,
            'teacher_loss': total_teacher_loss / num_batches,
            'student_loss': total_student_loss / num_batches,
            'distill_loss': total_distill_loss / num_batches,
            'feature_align_loss': total_feature_align_loss / num_batches,
            'common_sim_loss': total_common_sim_loss / num_batches,
            'private_diff_loss': total_private_diff_loss / num_batches,
            'teacher_acc': teacher_acc,  # 新增
            'student_acc': student_acc  # 新增
        }

    def validate_epoch(self, val_loader):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        total_teacher_loss = 0
        total_student_loss = 0
        total_distill_loss = 0
        total_feature_align_loss = 0
        total_common_sim_loss = 0
        total_private_diff_loss = 0

        # 计算准确率
        teacher_correct = 0
        student_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, (left_img, right_img, left_eeg, right_eeg, labels) in enumerate(val_loader):
                # 移动到设备
                left_img = left_img.to(self.device)
                right_img = right_img.to(self.device)
                left_eeg = left_eeg.to(self.device)
                right_eeg = right_eeg.to(self.device)
                labels = labels.to(self.device)

                # 前向传播
                outputs = self.model(
                    eeg1=left_eeg, eeg2=right_eeg,
                    img1=left_img, img2=right_img,
                    mode='train'
                )

                # 计算损失
                loss_dict = self.model.compute_loss(outputs, labels)
                loss = loss_dict['total_loss']

                # 记录所有损失
                total_loss += loss.item()
                total_teacher_loss += loss_dict['teacher_loss'].item()
                total_student_loss += loss_dict['student_loss'].item()
                total_distill_loss += loss_dict['distill_loss'].item()
                total_feature_align_loss += loss_dict['feature_align_loss'].item()
                total_common_sim_loss += loss_dict['common_sim_loss'].item()
                total_private_diff_loss += loss_dict['private_diff_loss'].item()

                # 计算准确率
                teacher_logits = outputs['teacher_logits']
                student_logits = outputs['student_logits']

                if teacher_logits.dim() == 1:  # 二分类
                    teacher_preds = (torch.sigmoid(teacher_logits) > 0.5).float()
                    student_preds = (torch.sigmoid(student_logits) > 0.5).float()
                else:  # 多分类
                    teacher_preds = torch.argmax(teacher_logits, dim=1)
                    student_preds = torch.argmax(student_logits, dim=1)

                teacher_correct += (teacher_preds == labels).sum().item()
                student_correct += (student_preds == labels).sum().item()
                total_samples += labels.size(0)

        # 计算平均指标
        num_batches = len(val_loader)
        teacher_acc = teacher_correct / total_samples * 100
        student_acc = student_correct / total_samples * 100

        return {
            'total_loss': total_loss / num_batches,
            'teacher_loss': total_teacher_loss / num_batches,
            'student_loss': total_student_loss / num_batches,
            'distill_loss': total_distill_loss / num_batches,
            'feature_align_loss': total_feature_align_loss / num_batches,
            'common_sim_loss': total_common_sim_loss / num_batches,
            'private_diff_loss': total_private_diff_loss / num_batches,
            'teacher_acc': teacher_acc,
            'student_acc': student_acc
        }

    def _save_training_history(self):
        """保存训练历史到JSON文件"""
        history = {
            'epochs': list(range(1, self.current_epoch + 1)),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_teacher_losses': self.train_teacher_losses,
            'train_student_losses': self.train_student_losses,
            'train_distill_losses': self.train_distill_losses,
            'val_teacher_losses': self.val_teacher_losses,
            'val_student_losses': self.val_student_losses,
            'val_distill_losses': self.val_distill_losses,
            'train_teacher_accuracies': self.train_teacher_accuracies,  # 新增
            'train_student_accuracies': self.train_student_accuracies,  # 新增
            'val_teacher_accuracies': self.val_teacher_accuracies,
            'val_student_accuracies': self.val_student_accuracies,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'best_teacher_acc': self.best_teacher_acc,
            'best_student_acc': self.best_student_acc,
            'end_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        history_path = os.path.join(self.output_dir, "training_history.json")
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        print("训练历史已保存")

    def _plot_training_results(self):
        """绘制训练结果图表 - 按照您的要求修改"""
        epochs = range(1, len(self.train_losses) + 1)

        # 创建图表 - 修改为1行3列的布局
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Privileged Learning Training Results', fontsize=16)

        # 1. 教师网络准确率曲线（训练和测试）
        axes[0].plot(epochs, self.train_teacher_accuracies, 'r-', label='Train Teacher Acc', alpha=0.7, linewidth=2)
        axes[0].plot(epochs, self.val_teacher_accuracies, 'b-', label='Val Teacher Acc', alpha=0.7, linewidth=2)
        axes[0].set_title('Teacher Network Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 100])  # 准确率范围0-100%

        # 2. 学生网络准确率曲线（训练和测试）
        axes[1].plot(epochs, self.train_student_accuracies, 'r-', label='Train Student Acc', alpha=0.7, linewidth=2)
        axes[1].plot(epochs, self.val_student_accuracies, 'g-', label='Val Student Acc', alpha=0.7, linewidth=2)
        axes[1].set_title('Student Network Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 100])  # 准确率范围0-100%

        # 3. 测试准确率曲线（教师和学生对比）
        axes[2].plot(epochs, self.val_teacher_accuracies, 'b-', label='Teacher Val Acc', alpha=0.7, linewidth=2)
        axes[2].plot(epochs, self.val_student_accuracies, 'g-', label='Student Val Acc', alpha=0.7, linewidth=2)
        axes[2].set_title('Teacher vs Student Validation Accuracy')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Accuracy (%)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([0, 100])  # 准确率范围0-100%

        plt.tight_layout()

        # 保存图表
        plot_path = os.path.join(self.output_dir, "training_accuracy_plots.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 额外保存一个包含所有损失曲线的图表
        self._plot_loss_curves(epochs)

        print(f"训练准确率图表已保存: {plot_path}")

    def _plot_loss_curves(self, epochs):
        """绘制损失曲线图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Privileged Learning Training Losses', fontsize=16)

        # 1. 总损失曲线
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train Loss', alpha=0.7)
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Val Loss', alpha=0.7)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 教师和学生损失曲线
        axes[0, 1].plot(epochs, self.train_teacher_losses, 'b-', label='Train Teacher', alpha=0.7)
        axes[0, 1].plot(epochs, self.train_student_losses, 'g-', label='Train Student', alpha=0.7)
        axes[0, 1].plot(epochs, self.val_teacher_losses, 'r-', label='Val Teacher', alpha=0.7)
        axes[0, 1].plot(epochs, self.val_student_losses, 'orange', label='Val Student', alpha=0.7)
        axes[0, 1].set_title('Teacher & Student Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 蒸馏损失曲线
        axes[1, 0].plot(epochs, self.train_distill_losses, 'b-', label='Train Distill', alpha=0.7)
        axes[1, 0].plot(epochs, self.val_distill_losses, 'r-', label='Val Distill', alpha=0.7)
        axes[1, 0].set_title('Distillation Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 学习率曲线
        axes[1, 1].plot(epochs, self.learning_rates, 'purple', alpha=0.7)
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')

        plt.tight_layout()

        # 保存损失图表
        loss_plot_path = os.path.join(self.output_dir, "training_loss_plots.png")
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"训练损失图表已保存: {loss_plot_path}")

    def train(self):
        """完整的训练流程"""
        print("开始训练特权学习网络...")
        start_time = time.time()

        # 恢复训练
        if self.config.resume_training and self.config.checkpoint_path:
            self.load_checkpoint(self.config.checkpoint_path)
            print(f"从 epoch {self.current_epoch} 恢复训练")

        # 创建数据加载器
        train_loader, val_loader = self.create_dataloaders()

        # 训练循环
        for epoch in range(self.current_epoch + 1, self.config.epochs + 1):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch}/{self.config.epochs}")
            print("-" * 50)

            # 训练
            epoch_start = time.time()
            train_metrics = self.train_epoch(train_loader)
            epoch_time = time.time() - epoch_start

            # 验证
            val_metrics = self.validate_epoch(val_loader)

            # 更新学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)

            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['total_loss'])
                else:
                    self.scheduler.step()

            # 记录所有损失和指标
            self.train_losses.append(train_metrics['total_loss'])
            self.val_losses.append(val_metrics['total_loss'])
            self.train_teacher_losses.append(train_metrics['teacher_loss'])
            self.train_student_losses.append(train_metrics['student_loss'])
            self.train_distill_losses.append(train_metrics['distill_loss'])
            self.train_feature_align_losses.append(train_metrics['feature_align_loss'])
            self.train_common_sim_losses.append(train_metrics['common_sim_loss'])
            self.train_private_diff_losses.append(train_metrics['private_diff_loss'])
            self.val_teacher_losses.append(val_metrics['teacher_loss'])
            self.val_student_losses.append(val_metrics['student_loss'])
            self.val_distill_losses.append(val_metrics['distill_loss'])
            self.val_feature_align_losses.append(val_metrics['feature_align_loss'])
            self.val_common_sim_losses.append(val_metrics['common_sim_loss'])
            self.val_private_diff_losses.append(val_metrics['private_diff_loss'])

            # 记录准确率 - 新增训练准确率
            self.train_teacher_accuracies.append(train_metrics['teacher_acc'])
            self.train_student_accuracies.append(train_metrics['student_acc'])
            self.val_teacher_accuracies.append(val_metrics['teacher_acc'])
            self.val_student_accuracies.append(val_metrics['student_acc'])

            # 更新最佳指标
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
            if val_metrics['teacher_acc'] > self.best_teacher_acc:
                self.best_teacher_acc = val_metrics['teacher_acc']
            if val_metrics['student_acc'] > self.best_student_acc:
                self.best_student_acc = val_metrics['student_acc']

            # 打印所有指标 - 详细格式
            print(f"\n📊 Epoch {epoch} 训练结果:")
            print(f"  学习率: {current_lr:.2e}")
            print(f"  🎯 训练损失:")
            print(f"    • 总损失: {train_metrics['total_loss']:.6f}")
            print(f"    • 教师损失: {train_metrics['teacher_loss']:.6f}")
            print(f"    • 学生损失: {train_metrics['student_loss']:.6f}")
            print(f"    • 蒸馏损失: {train_metrics['distill_loss']:.6f}")
            print(f"  📈 训练准确率:")
            print(f"    • 教师: {train_metrics['teacher_acc']:.2f}%")
            print(f"    • 学生: {train_metrics['student_acc']:.2f}%")
            print(f"  🎯 验证损失:")
            print(f"    • 总损失: {val_metrics['total_loss']:.6f}")
            print(f"    • 教师损失: {val_metrics['teacher_loss']:.6f}")
            print(f"    • 学生损失: {val_metrics['student_loss']:.6f}")
            print(f"    • 蒸馏损失: {val_metrics['distill_loss']:.6f}")
            print(f"  📈 验证准确率:")
            print(f"    • 教师: {val_metrics['teacher_acc']:.2f}%")
            print(f"    • 学生: {val_metrics['student_acc']:.2f}%")
            print(f"  ⏱️  时间: {epoch_time:.2f}s")

            # 保存最佳模型
            if val_metrics['total_loss'] < self.best_val_loss:
                self.save_checkpoint(is_best=True)
                print("✅ 保存最佳模型")

            # 定期保存检查点
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(is_best=False)
                print("💾 保存检查点")

            # 定期保存训练历史和图表
            if epoch % self.config.save_interval == 0 or epoch == self.config.epochs:
                self._save_training_history()
                if epoch > 1:  # 至少有两个epoch才能画图
                    self._plot_training_results()

        # 训练完成
        total_time = time.time() - start_time
        print(f"\n训练完成! 总时间: {total_time:.2f}s")
        print(f"最佳验证损失: {self.best_val_loss:.6f}")
        print(f"最佳教师准确率: {self.best_teacher_acc:.2f}%")
        print(f"最佳学生准确率: {self.best_student_acc:.2f}%")

        # 保存最终模型和结果
        self.save_checkpoint(is_best=False, is_final=True)
        self._save_training_history()
        self._plot_training_results()

        return self.train_losses, self.val_losses

    def save_checkpoint(self, is_best=False, is_final=False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_teacher_losses': self.train_teacher_losses,
            'train_student_losses': self.train_student_losses,
            'train_distill_losses': self.train_distill_losses,
            'val_teacher_losses': self.val_teacher_losses,
            'val_student_losses': self.val_student_losses,
            'val_distill_losses': self.val_distill_losses,
            'train_teacher_accuracies': self.train_teacher_accuracies,  # 新增
            'train_student_accuracies': self.train_student_accuracies,  # 新增
            'val_teacher_accuracies': self.val_teacher_accuracies,
            'val_student_accuracies': self.val_student_accuracies,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'best_teacher_acc': self.best_teacher_acc,
            'best_student_acc': self.best_student_acc,
            'config': self.config.to_dict()
        }

        if is_best:
            filename = "privileged_best_model.pth"
        elif is_final:
            filename = f"privileged_final_model_epoch{self.current_epoch}.pth"
        else:
            filename = f"checkpoints/privileged_checkpoint_epoch{self.current_epoch}.pth"

        filepath = os.path.join(self.output_dir, filename)
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)

        # 同时保存学生网络的独立模型（用于推理）
        if is_best or is_final:
            student_model = self.model.module.student_network if hasattr(self.model,
                                                                         'module') else self.model.student_network
            student_checkpoint = {
                'student_state_dict': student_model.state_dict(),
                'config': self.config.to_dict()
            }
            student_filepath = os.path.join(self.output_dir, f"student_{filename}")
            torch.save(student_checkpoint, student_filepath)

    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # 恢复训练历史
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_teacher_losses = checkpoint.get('train_teacher_losses', [])
        self.train_student_losses = checkpoint.get('train_student_losses', [])
        self.train_distill_losses = checkpoint.get('train_distill_losses', [])
        self.val_teacher_losses = checkpoint.get('val_teacher_losses', [])
        self.val_student_losses = checkpoint.get('val_student_losses', [])
        self.val_distill_losses = checkpoint.get('val_distill_losses', [])
        self.train_teacher_accuracies = checkpoint.get('train_teacher_accuracies', [])  # 新增
        self.train_student_accuracies = checkpoint.get('train_student_accuracies', [])  # 新增
        self.val_teacher_accuracies = checkpoint.get('val_teacher_accuracies', [])
        self.val_student_accuracies = checkpoint.get('val_student_accuracies', [])
        self.learning_rates = checkpoint.get('learning_rates', [])

        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_teacher_acc = checkpoint.get('best_teacher_acc', 0.0)
        self.best_student_acc = checkpoint.get('best_student_acc', 0.0)
        self.current_epoch = checkpoint['epoch']

        print(f"加载检查点: epoch {self.current_epoch}, 最佳验证损失: {self.best_val_loss:.6f}")

        return self.current_epoch


def main():
    """主训练函数"""
    # 获取配置
    config = get_config('default')  # 可选: 'default', 'debug', 'large', 'small'

    # 创建训练器
    trainer = PrivilegedTrainer(config)

    # 开始训练
    train_losses, val_losses = trainer.train()

    print("\n训练完成!")
    print(f"最终训练损失: {train_losses[-1]:.6f}")
    print(f"最终验证损失: {val_losses[-1]:.6f}")
    print(f"模型保存在: {trainer.output_dir}")


if __name__ == "__main__":
    main()