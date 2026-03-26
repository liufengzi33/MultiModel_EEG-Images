import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import time
import warnings
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import f1_score, roc_auc_score

warnings.filterwarnings('ignore')

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

        # 创建模型 (强制设置学生网络为 eeg)
        self.model = self._create_model()

        # 优化器和损失函数
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # 历史记录字典 (整合所有指标)
        self.history = {
            'train_losses': [], 'val_losses': [],
            'train_teacher_losses': [], 'val_teacher_losses': [],
            'train_student_losses': [], 'val_student_losses': [],
            'train_distill_losses': [], 'val_distill_losses': [],
            'train_common_sim_losses': [], 'val_common_sim_losses': [],
            'train_private_diff_losses': [], 'val_private_diff_losses': [],

            'train_teacher_acc': [], 'val_teacher_acc': [],
            'train_student_acc': [], 'val_student_acc': [],

            'train_teacher_f1': [], 'val_teacher_f1': [],
            'train_student_f1': [], 'val_student_f1': [],

            'train_teacher_auc': [], 'val_teacher_auc': [],
            'train_student_auc': [], 'val_student_auc': [],

            'train_krr': [], 'val_krr': [],  # 知识保留率 Knowledge Retention Rate
            'learning_rates': [], 'temperatures': []
        }

        self.best_val_loss = float('inf')
        self.best_teacher_acc = 0.0
        self.best_student_acc = 0.0
        self.current_epoch = 0

    def _create_model(self):
        """创建特权学习模型"""
        model = PrivilegedMultimodalNetwork(
            student_modality=self.config.student_modality, # 用eeg作为学生网络
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
        if self.config.scheduler.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.epochs)
        elif self.config.scheduler.lower() == 'step':
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif self.config.scheduler.lower() == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=self.config.patience,
                                                        factor=self.config.factor, verbose=True)
        return None

    def create_dataloaders(self):
        print("创建数据加载器...")
        dataset = MyPP2Dataset(is_flipped=False, transform=self.config.transform, subject_id=self.config.subject_id)
        train_loader, val_loader = create_dataloaders(dataset=dataset, batch_size=self.config.batch_size, shuffle=True)
        return train_loader, val_loader

    def _update_temperature(self, epoch):
        """改进的温度退火策略：指数衰减 (Exponential Decay)"""
        start_temp = self.config.temperature
        end_temp = 1.0  # 最终温度收敛到 1.0
        decay_rate = 0.97  # 衰减率，可根据需要调整。0.95 意味着每轮变为上一轮的 95%

        # 指数计算当前温度
        current_temp = start_temp * (decay_rate ** epoch)

        # 保证温度不会低于 end_temp
        self.model.temperature = max(current_temp, end_temp)
        return self.model.temperature

    def train_epoch(self, train_loader):
        self.model.train()
        metrics = {k: 0.0 for k in ['total_loss', 'teacher_loss', 'student_loss', 'distill_loss', 'common_sim_loss',
                                    'private_diff_loss']}

        all_labels, all_t_preds, all_s_preds = [], [], []
        all_t_probs, all_s_probs = [], []

        for left_img, right_img, left_eeg, right_eeg, labels in train_loader:
            left_img, right_img = left_img.to(self.device), right_img.to(self.device)
            left_eeg, right_eeg = left_eeg.to(self.device), right_eeg.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(eeg1=left_eeg, eeg2=right_eeg, img1=left_img, img2=right_img, mode='train')
            loss_dict = self.model.compute_loss(outputs, labels)

            loss = loss_dict['total_loss']
            loss.backward()

            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            # 累加损失
            for k in metrics.keys():
                metrics[k] += loss_dict[k].item()

            # 提取概率和预测结果
            t_logits, s_logits = outputs['teacher_logits'], outputs['student_logits']
            if t_logits.dim() == 1:
                t_probs, s_probs = torch.sigmoid(t_logits), torch.sigmoid(s_logits)
                t_preds, s_preds = (t_probs > 0.5).long(), (s_probs > 0.5).long()
            else:
                t_probs, s_probs = F.softmax(t_logits, dim=1)[:, 1], F.softmax(s_logits, dim=1)[:, 1]
                t_preds, s_preds = torch.argmax(t_logits, dim=1), torch.argmax(s_logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_t_preds.extend(t_preds.cpu().numpy())
            all_s_preds.extend(s_preds.cpu().numpy())
            all_t_probs.extend(t_probs.detach().cpu().numpy())
            all_s_probs.extend(s_probs.detach().cpu().numpy())

        num_batches = len(train_loader)
        metrics = {k: v / num_batches for k, v in metrics.items()}

        # 计算 Acc, F1, AUC
        from sklearn.metrics import accuracy_score
        metrics['teacher_acc'] = accuracy_score(all_labels, all_t_preds) * 100
        metrics['student_acc'] = accuracy_score(all_labels, all_s_preds) * 100
        metrics['teacher_f1'] = f1_score(all_labels, all_t_preds, average='macro')
        metrics['student_f1'] = f1_score(all_labels, all_s_preds, average='macro')
        try:
            metrics['teacher_auc'] = roc_auc_score(all_labels, all_t_probs)
            metrics['student_auc'] = roc_auc_score(all_labels, all_s_probs)
        except ValueError:
            metrics['teacher_auc'], metrics['student_auc'] = 0.5, 0.5

        # 核心要求3：计算知识保留率 KRR
        metrics['krr'] = (metrics['student_acc'] / metrics['teacher_acc'] * 100) if metrics['teacher_acc'] > 0 else 0.0

        return metrics

    def validate_epoch(self, val_loader):
        self.model.eval()
        metrics = {k: 0.0 for k in ['total_loss', 'teacher_loss', 'student_loss', 'distill_loss', 'common_sim_loss',
                                    'private_diff_loss']}

        all_labels, all_t_preds, all_s_preds = [], [], []
        all_t_probs, all_s_probs = [], []

        with torch.no_grad():
            for left_img, right_img, left_eeg, right_eeg, labels in val_loader:
                left_img, right_img = left_img.to(self.device), right_img.to(self.device)
                left_eeg, right_eeg = left_eeg.to(self.device), right_eeg.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(eeg1=left_eeg, eeg2=right_eeg, img1=left_img, img2=right_img, mode='train')
                loss_dict = self.model.compute_loss(outputs, labels)

                for k in metrics.keys():
                    metrics[k] += loss_dict[k].item()

                t_logits, s_logits = outputs['teacher_logits'], outputs['student_logits']
                if t_logits.dim() == 1:
                    t_probs, s_probs = torch.sigmoid(t_logits), torch.sigmoid(s_logits)
                    t_preds, s_preds = (t_probs > 0.5).long(), (s_probs > 0.5).long()
                else:
                    t_probs, s_probs = F.softmax(t_logits, dim=1)[:, 1], F.softmax(s_logits, dim=1)[:, 1]
                    t_preds, s_preds = torch.argmax(t_logits, dim=1), torch.argmax(s_logits, dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_t_preds.extend(t_preds.cpu().numpy())
                all_s_preds.extend(s_preds.cpu().numpy())
                all_t_probs.extend(t_probs.cpu().numpy())
                all_s_probs.extend(s_probs.cpu().numpy())

        num_batches = len(val_loader)
        metrics = {k: v / num_batches for k, v in metrics.items()}

        from sklearn.metrics import accuracy_score
        metrics['teacher_acc'] = accuracy_score(all_labels, all_t_preds) * 100
        metrics['student_acc'] = accuracy_score(all_labels, all_s_preds) * 100
        metrics['teacher_f1'] = f1_score(all_labels, all_t_preds, average='macro')
        metrics['student_f1'] = f1_score(all_labels, all_s_preds, average='macro')
        try:
            metrics['teacher_auc'] = roc_auc_score(all_labels, all_t_probs)
            metrics['student_auc'] = roc_auc_score(all_labels, all_s_probs)
        except ValueError:
            metrics['teacher_auc'], metrics['student_auc'] = 0.5, 0.5

        # 计算验证集的 KRR
        metrics['krr'] = (metrics['student_acc'] / metrics['teacher_acc'] * 100) if metrics['teacher_acc'] > 0 else 0.0

        return metrics

    def _save_training_history(self):
        """保存训练历史到JSON文件"""
        history_save = self.history.copy()
        history_save['epochs'] = list(range(1, self.current_epoch + 1))
        history_save['best_val_loss'] = self.best_val_loss
        history_save['best_teacher_acc'] = self.best_teacher_acc
        history_save['best_student_acc'] = self.best_student_acc
        history_save['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        history_path = os.path.join(self.output_dir, "training_history.json")
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history_save, f, indent=2, ensure_ascii=False)

    def _plot_training_results(self):
        """核心要求4：绘制学术风格 (Seaborn) 的四张图"""
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        epochs = range(1, len(self.history['train_losses']) + 1)
        model_name = f"Privileged (Student: {self.model.student_modality.upper()})"

        # ==========================================
        # 图 1: 损失大图 (2x2)
        # ==========================================
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # 1. 总损失
        axes[0, 0].plot(epochs, self.history['train_losses'], marker='o', markersize=4, label='Train Loss', alpha=0.8)
        axes[0, 0].plot(epochs, self.history['val_losses'], marker='s', markersize=4, label='Val Loss', alpha=0.8)
        axes[0, 0].set_title('Total Loss', fontweight='bold')
        axes[0, 0].legend(loc='upper right', frameon=True, shadow=True)

        # 2. 教师与学生损失
        axes[0, 1].plot(epochs, self.history['train_teacher_losses'], marker='o', markersize=4,
                        label='Train Teacher Loss', alpha=0.8)
        axes[0, 1].plot(epochs, self.history['train_student_losses'], marker='^', markersize=4,
                        label='Train Student Loss', alpha=0.8)
        axes[0, 1].plot(epochs, self.history['val_teacher_losses'], marker='s', markersize=4, label='Val Teacher Loss',
                        alpha=0.8)
        axes[0, 1].plot(epochs, self.history['val_student_losses'], marker='v', markersize=4, label='Val Student Loss',
                        alpha=0.8)
        axes[0, 1].set_title('Teacher vs Student Loss', fontweight='bold')
        axes[0, 1].legend(loc='upper right', frameon=True, shadow=True)

        # 3. 蒸馏损失 (Distillation Loss)
        axes[1, 0].plot(epochs, self.history['train_distill_losses'], marker='o', markersize=4,
                        label='Train Distill Loss', alpha=0.8, color='purple')
        axes[1, 0].plot(epochs, self.history['val_distill_losses'], marker='s', markersize=4, label='Val Distill Loss',
                        alpha=0.8, color='violet')
        axes[1, 0].set_title('Distillation Loss', fontweight='bold')
        axes[1, 0].legend(loc='upper right', frameon=True, shadow=True)

        # 4. 对齐差异损失
        axes[1, 1].plot(epochs, self.history['train_common_sim_losses'], marker='o', markersize=4,
                        label='Train Common Sim Loss', alpha=0.8, color='forestgreen')
        axes[1, 1].plot(epochs, self.history['train_private_diff_losses'], marker='^', markersize=4,
                        label='Train Private Diff Loss', alpha=0.8, color='darkorange')
        axes[1, 1].set_title('Feature Alignment & Difference Loss', fontweight='bold')
        axes[1, 1].legend(loc='upper right', frameon=True, shadow=True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'all_losses_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # ==========================================
        # 图 2: 准确率曲线 (Teacher vs Student)
        # ==========================================
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, self.history['train_teacher_acc'], marker='o', markersize=4, label='Train Teacher Acc',
                 alpha=0.8)
        plt.plot(epochs, self.history['val_teacher_acc'], marker='s', markersize=4, label='Val Teacher Acc', alpha=0.8)
        plt.plot(epochs, self.history['train_student_acc'], marker='^', markersize=4, label='Train Student Acc',
                 alpha=0.8)
        plt.plot(epochs, self.history['val_student_acc'], marker='v', markersize=4, label='Val Student Acc', alpha=0.8)
        plt.title(f'{model_name} - Accuracy', fontweight='bold')
        plt.xlabel('Epoch', fontweight='bold')
        plt.ylabel('Accuracy (%)', fontweight='bold')
        plt.legend(loc='lower right', frameon=True, shadow=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'accuracy_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # ==========================================
        # 图 3: F1 分数曲线
        # ==========================================
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, self.history['train_teacher_f1'], marker='o', markersize=4, label='Train Teacher F1',
                 alpha=0.8)
        plt.plot(epochs, self.history['val_teacher_f1'], marker='s', markersize=4, label='Val Teacher F1', alpha=0.8)
        plt.plot(epochs, self.history['train_student_f1'], marker='^', markersize=4, label='Train Student F1',
                 alpha=0.8)
        plt.plot(epochs, self.history['val_student_f1'], marker='v', markersize=4, label='Val Student F1', alpha=0.8)
        plt.title(f'{model_name} - F1 Score', fontweight='bold')
        plt.xlabel('Epoch', fontweight='bold')
        plt.ylabel('F1 Score', fontweight='bold')
        plt.legend(loc='lower right', frameon=True, shadow=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'f1_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # ==========================================
        # 图 4: AUC-ROC 曲线
        # ==========================================
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, self.history['train_teacher_auc'], marker='o', markersize=4, label='Train Teacher AUC',
                 alpha=0.8)
        plt.plot(epochs, self.history['val_teacher_auc'], marker='s', markersize=4, label='Val Teacher AUC', alpha=0.8)
        plt.plot(epochs, self.history['train_student_auc'], marker='^', markersize=4, label='Train Student AUC',
                 alpha=0.8)
        plt.plot(epochs, self.history['val_student_auc'], marker='v', markersize=4, label='Val Student AUC', alpha=0.8)
        plt.title(f'{model_name} - AUC-ROC', fontweight='bold')
        plt.xlabel('Epoch', fontweight='bold')
        plt.ylabel('AUC-ROC', fontweight='bold')
        plt.legend(loc='lower right', frameon=True, shadow=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'auc_roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

        sns.reset_orig()
        print(f"✅ 图表保存完成！")

    def train(self):
        print("开始训练特权学习网络...")
        start_time = time.time()

        if self.config.resume_training and self.config.checkpoint_path:
            self.load_checkpoint(self.config.checkpoint_path)
            print(f"从 epoch {self.current_epoch} 恢复训练")

        train_loader, val_loader = self.create_dataloaders()

        for epoch in range(self.current_epoch + 1, self.config.epochs + 1):
            self.current_epoch = epoch

            # 调整温度并记录
            current_temp = self._update_temperature(epoch)
            self.history['temperatures'].append(current_temp)

            print(f"\nEpoch {epoch}/{self.config.epochs} [Temp: {current_temp:.2f}]")
            print("-" * 50)

            # 训练
            epoch_start = time.time()
            train_metrics = self.train_epoch(train_loader)
            epoch_time = time.time() - epoch_start

            # 验证
            val_metrics = self.validate_epoch(val_loader)

            # 更新学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)

            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['total_loss'])
                else:
                    self.scheduler.step()

            # 记录历史
            for split, metrics in [('train', train_metrics), ('val', val_metrics)]:
                self.history[f'{split}_losses'].append(metrics['total_loss'])
                self.history[f'{split}_teacher_losses'].append(metrics['teacher_loss'])
                self.history[f'{split}_student_losses'].append(metrics['student_loss'])
                self.history[f'{split}_distill_losses'].append(metrics['distill_loss'])
                self.history[f'{split}_common_sim_losses'].append(metrics['common_sim_loss'])
                self.history[f'{split}_private_diff_losses'].append(metrics['private_diff_loss'])
                self.history[f'{split}_teacher_acc'].append(metrics['teacher_acc'])
                self.history[f'{split}_student_acc'].append(metrics['student_acc'])
                self.history[f'{split}_teacher_f1'].append(metrics['teacher_f1'])
                self.history[f'{split}_student_f1'].append(metrics['student_f1'])
                self.history[f'{split}_teacher_auc'].append(metrics['teacher_auc'])
                self.history[f'{split}_student_auc'].append(metrics['student_auc'])
                self.history[f'{split}_krr'].append(metrics['krr'])

            # 更新最佳指标
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
            if val_metrics['teacher_acc'] > self.best_teacher_acc:
                self.best_teacher_acc = val_metrics['teacher_acc']
            if val_metrics['student_acc'] > self.best_student_acc:
                self.best_student_acc = val_metrics['student_acc']

            # 打印详细结果
            print(f"📊 Epoch {epoch} 总结:")
            print(f"  学习率: {current_lr:.2e}")
            print(f"  🎯 损失 (Train/Val): 总 ({train_metrics['total_loss']:.4f}/{val_metrics['total_loss']:.4f}) | "
                  f"蒸馏 ({train_metrics['distill_loss']:.4f}/{val_metrics['distill_loss']:.4f})")
            print(
                f"  📈 准确率 Acc (Train/Val): 教师 ({train_metrics['teacher_acc']:.2f}%/{val_metrics['teacher_acc']:.2f}%) | "
                f"学生 ({train_metrics['student_acc']:.2f}%/{val_metrics['student_acc']:.2f}%)")
            print(
                f"  📈 F1 分数 (Train/Val): 教师 ({train_metrics['teacher_f1']:.4f}/{val_metrics['teacher_f1']:.4f}) | "
                f"学生 ({train_metrics['student_f1']:.4f}/{val_metrics['student_f1']:.4f})")
            print(
                f"  📈 AUC-ROC (Train/Val): 教师 ({train_metrics['teacher_auc']:.4f}/{val_metrics['teacher_auc']:.4f}) | "
                f"学生 ({train_metrics['student_auc']:.4f}/{val_metrics['student_auc']:.4f})")
            print(f"  🧠 KRR (知识保留率): 训练 {train_metrics['krr']:.2f}% | 验证 {val_metrics['krr']:.2f}%")
            print(f"  ⏱️  用时: {epoch_time:.2f}s")

            # 保存
            if val_metrics['total_loss'] <= self.best_val_loss:
                self.save_checkpoint(is_best=True)
                print("  ✅ 保存最佳模型")

            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(is_best=False)

            if epoch % self.config.save_interval == 0 or epoch == self.config.epochs:
                self._save_training_history()
                if epoch > 1:
                    self._plot_training_results()

        print(f"\n训练完成! 总时间: {time.time() - start_time:.2f}s")
        print(f"最佳验证损失: {self.best_val_loss:.6f}")
        print(f"最佳教师/学生准确率: {self.best_teacher_acc:.2f}% / {self.best_student_acc:.2f}%")

        self.save_checkpoint(is_best=False, is_final=True)
        self._save_training_history()
        self._plot_training_results()

        return self.history['train_losses'], self.history['val_losses']

    def save_checkpoint(self, is_best=False, is_final=False):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_teacher_acc': self.best_teacher_acc,
            'best_student_acc': self.best_student_acc,
            'config': self.config.to_dict()
        }

        filename = "privileged_best_model.pth" if is_best else (
            f"privileged_final_model_epoch{self.current_epoch}.pth" if is_final else f"checkpoints/privileged_checkpoint_epoch{self.current_epoch}.pth")
        filepath = os.path.join(self.output_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)

        if is_best or is_final:
            student_model = self.model.module.student_network if hasattr(self.model,
                                                                         'module') else self.model.student_network
            torch.save({'student_state_dict': student_model.state_dict(), 'config': self.config.to_dict()},
                       os.path.join(self.output_dir, f"student_{filename}"))

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.history = checkpoint.get('history', self.history)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_teacher_acc = checkpoint.get('best_teacher_acc', 0.0)
        self.best_student_acc = checkpoint.get('best_student_acc', 0.0)
        self.current_epoch = checkpoint['epoch']

        return self.current_epoch


def main():
    config = get_config('default')
    trainer = PrivilegedTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()