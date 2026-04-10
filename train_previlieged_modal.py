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
from tqdm import tqdm

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
        self.config.output_dir = self.output_dir
        print(f"输出目录: {self.output_dir}")

        # 创建模型
        self.model = self._create_model()

        # 历史记录字典 (扁平化连续记录，方便跨阶段无缝绘图)
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

            'train_krr': [], 'val_krr': [],  # 知识保留率
            'learning_rates': [], 'temperatures': []
        }

        self.best_val_loss = float('inf')
        self.best_teacher_acc = 0.0
        self.best_student_acc = 0.0
        self.global_epoch = 0  # 全局 Epoch 计数器，用于温度衰减

    def _create_model(self):
        """创建特权学习模型"""
        model = PrivilegedMultimodalNetwork(
            student_modality=self.config.student_modality,
            eeg_model_name=self.config.eeg_model_name,
            image_model_name=self.config.image_model_name,
            image_model_type=self.config.image_model_type,
            in_chans=self.config.in_chans,
            n_classes=self.config.n_classes,
            input_window_samples=self.config.input_window_samples,
            use_pretrained_eeg=self.config.use_pretrained,
            use_pretrained_image=self.config.use_pretrained,
            base_path=self.config.base_path,
            feature_dim=self.config.feature_dim,
            dropout_rate=self.config.dropout_rate,
            alpha=self.config.alpha,
            beta=self.config.beta,
            gamma=self.config.gamma,
            temperature=self.config.temperature
        )
        return model.to(self.device)

    def create_dataloaders(self):
        print("📦 创建数据加载器...")
        dataset = MyPP2Dataset(is_flipped=False, transform=self.config.transform, subject_id=self.config.subject_id)
        train_loader, val_loader = create_dataloaders(dataset=dataset, batch_size=self.config.batch_size, shuffle=True)
        return train_loader, val_loader

    # ==================== 冻结与解冻逻辑 (三阶段核心) ====================
    def freeze_backbones(self):
        """冻结教师和学生网络的所有 Backbone 参数"""
        # 冻结教师网络特征提取层
        for p in self.model.teacher_network['eeg_path'].parameters(): p.requires_grad = False
        for p in self.model.teacher_network['image_path'].parameters(): p.requires_grad = False
        # 冻结学生网络特征提取层
        for p in self.model.student_network['feature_path'].parameters(): p.requires_grad = False
        print("🔒 冻结教师与学生网络的所有 Backbone 参数")

    def _unfreeze_eeg_path(self, path_module, layers):
        """安全地解冻 EEG 通路最后几层 (智能适配层级)"""
        target_module = None
        # 兼容不同封装寻找真正的 EEG特征层
        if hasattr(path_module, 'feature_extractor') and hasattr(path_module.feature_extractor, 'feature_net'):
            target_module = path_module.feature_extractor.feature_net
        elif hasattr(path_module, 'feature_net'):
            target_module = path_module.feature_net
        else:
            target_module = path_module

        if target_module is not None:
            modules = list(target_module.children())
            if len(modules) > 0:
                unfreeze_indices = list(range(len(modules)))[-layers:]
                for idx in unfreeze_indices:
                    for p in modules[idx].parameters(): p.requires_grad = True

        # 保证融合层被解冻
        if hasattr(path_module, 'fusion') and path_module.fusion is not None:
            for p in path_module.fusion.parameters(): p.requires_grad = True

    def _unfreeze_img_path(self, path_module, layers):
        """安全地解冻 Image 通路最后几层 (智能适配层级)"""
        target_module = None

        # 1. 动态向下寻找包含 CNN 卷积层的对象
        if hasattr(path_module, 'feature_extractor') and hasattr(path_module.feature_extractor, 'features'):
            target_module = path_module.feature_extractor.features
        elif hasattr(path_module, 'base_model') and hasattr(path_module.base_model, 'features'):
            target_module = path_module.base_model.features
        elif hasattr(path_module, 'features'):
            target_module = path_module.features
        else:
            # 智能后备：获取参数最多的那个子模块（通常就是主干CNN）
            target_module = max(path_module.children(), key=lambda m: sum(p.numel() for p in m.parameters()),
                                default=path_module)
            if hasattr(target_module, 'features'):
                target_module = target_module.features

        # 2. 按层数解冻
        if target_module is not None:
            modules = list(target_module.children())
            if len(modules) > 0:
                unfreeze_indices = list(range(len(modules)))[-layers:]
                for idx in unfreeze_indices:
                    for p in modules[idx].parameters(): p.requires_grad = True
            else:
                # 极端后备方案：直接拿最后的几个 parameter 张量解冻
                params = list(target_module.parameters())
                for p in params[-layers * 2:]:
                    p.requires_grad = True

        # 3. 保证融合层被解冻
        if hasattr(path_module, 'fusion_convs') and path_module.fusion_convs is not None:
            for p in path_module.fusion_convs.parameters(): p.requires_grad = True
        elif hasattr(path_module, 'fusion') and path_module.fusion is not None:
            for p in path_module.fusion.parameters(): p.requires_grad = True

    def unfreeze_backbone_last_layers(self, eeg_layers=4, img_layers=4):
        """解冻 Backbone 最后几层"""
        # 教师网络
        self._unfreeze_eeg_path(self.model.teacher_network['eeg_path'], eeg_layers)
        self._unfreeze_img_path(self.model.teacher_network['image_path'], img_layers)

        # 学生网络 (根据模态决定)
        if self.model.student_modality == 'eeg':
            self._unfreeze_eeg_path(self.model.student_network['feature_path'], eeg_layers)
        else:
            self._unfreeze_img_path(self.model.student_network['feature_path'], img_layers)

        print(f"🔓 解冻 Backbone 最后几层 (EEG: {eeg_layers}层, Image: {img_layers}层)")

    def unfreeze_heads(self):
        """解冻所有分类头、编码器和私有/公共映射层"""
        for k, v in self.model.teacher_network.items():
            if 'encoder' in k or 'classifier' in k:
                for p in v.parameters(): p.requires_grad = True

        for k, v in self.model.student_network.items():
            if 'encoder' in k or 'classifier' in k:
                for p in v.parameters(): p.requires_grad = True
        print("🔓 解冻 Head 投影网络和分类器")

    def build_optimizer(self, lr_head, lr_backbone):
        """构建分组优化器，给予 Backbone 极小学习率，Head 正规学习率"""
        head_params = []
        backbone_params = []

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if 'path' in name:  # 匹配 eeg_path, image_path, feature_path (均为 Backbone)
                backbone_params.append(p)
            else:
                head_params.append(p)

        param_groups = []
        if head_params:
            param_groups.append({'params': head_params, 'lr': lr_head})
        if backbone_params:
            param_groups.append({'params': backbone_params, 'lr': lr_backbone})

        optimizer = optim.AdamW(param_groups, weight_decay=self.config.weight_decay)
        print(
            f"🔧 优化器准备完毕: Head LR={lr_head:.2e} ({len(head_params)}组), Backbone LR={lr_backbone:.2e} ({len(backbone_params)}组)")
        return optimizer

    # ==================== 温度更新与指标计算 ====================
    def _update_temperature(self):
        start_temp = self.config.temperature
        end_temp = 1.0
        decay_rate = 0.96
        current_temp = start_temp * (decay_rate ** self.global_epoch)
        self.model.temperature = max(current_temp, end_temp)
        return self.model.temperature

    def _compute_metrics_from_outputs(self, outputs, labels, loss_dict, metrics_accum, arrays_dict):
        """辅助方法：提取预测概率并累加 Loss"""
        for k in metrics_accum.keys():
            metrics_accum[k] += loss_dict[k].item()

        t_logits, s_logits = outputs['teacher_logits'], outputs['student_logits']
        if t_logits.dim() == 1:
            t_probs, s_probs = torch.sigmoid(t_logits), torch.sigmoid(s_logits)
            t_preds, s_preds = (t_probs > 0.5).long(), (s_probs > 0.5).long()
        else:
            t_probs, s_probs = F.softmax(t_logits, dim=1)[:, 1], F.softmax(s_logits, dim=1)[:, 1]
            t_preds, s_preds = torch.argmax(t_logits, dim=1), torch.argmax(s_logits, dim=1)

        arrays_dict['labels'].extend(labels.cpu().numpy())
        arrays_dict['t_preds'].extend(t_preds.cpu().numpy())
        arrays_dict['s_preds'].extend(s_preds.cpu().numpy())
        arrays_dict['t_probs'].extend(t_probs.detach().cpu().numpy())
        arrays_dict['s_probs'].extend(s_probs.detach().cpu().numpy())

    def _calculate_final_metrics(self, metrics_accum, arrays_dict, num_batches):
        """辅助方法：计算 Epoch 的最终 Acc/F1/AUC/KRR"""
        metrics = {k: v / num_batches for k, v in metrics_accum.items()}
        from sklearn.metrics import accuracy_score

        metrics['teacher_acc'] = accuracy_score(arrays_dict['labels'], arrays_dict['t_preds']) * 100
        metrics['student_acc'] = accuracy_score(arrays_dict['labels'], arrays_dict['s_preds']) * 100
        metrics['teacher_f1'] = f1_score(arrays_dict['labels'], arrays_dict['t_preds'], average='macro')
        metrics['student_f1'] = f1_score(arrays_dict['labels'], arrays_dict['s_preds'], average='macro')
        try:
            metrics['teacher_auc'] = roc_auc_score(arrays_dict['labels'], arrays_dict['t_probs'])
            metrics['student_auc'] = roc_auc_score(arrays_dict['labels'], arrays_dict['s_probs'])
        except ValueError:
            metrics['teacher_auc'], metrics['student_auc'] = 0.5, 0.5

        metrics['krr'] = (metrics['student_acc'] / metrics['teacher_acc'] * 100) if metrics['teacher_acc'] > 0 else 0.0
        return metrics

    # ==================== 训练与验证核心逻辑 ====================
    def train_epoch(self, train_loader, optimizer):
        self.model.train()
        metrics_accum = {k: 0.0 for k in
                         ['total_loss', 'teacher_loss', 'student_loss', 'distill_loss', 'common_sim_loss',
                          'private_diff_loss']}
        arrays = {'labels': [], 't_preds': [], 's_preds': [], 't_probs': [], 's_probs': []}

        for left_img, right_img, left_eeg, right_eeg, labels in train_loader:
            left_img, right_img, left_eeg, right_eeg, labels = (
                left_img.to(self.device), right_img.to(self.device),
                left_eeg.to(self.device), right_eeg.to(self.device), labels.to(self.device)
            )

            optimizer.zero_grad()
            outputs = self.model(eeg1=left_eeg, eeg2=right_eeg, img1=left_img, img2=right_img, mode='train')
            loss_dict = self.model.compute_loss(outputs, labels)

            loss_dict['total_loss'].backward()
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            optimizer.step()

            self._compute_metrics_from_outputs(outputs, labels, loss_dict, metrics_accum, arrays)

        return self._calculate_final_metrics(metrics_accum, arrays, len(train_loader))

    def validate_epoch(self, val_loader):
        self.model.eval()
        metrics_accum = {k: 0.0 for k in
                         ['total_loss', 'teacher_loss', 'student_loss', 'distill_loss', 'common_sim_loss',
                          'private_diff_loss']}
        arrays = {'labels': [], 't_preds': [], 's_preds': [], 't_probs': [], 's_probs': []}

        with torch.no_grad():
            for left_img, right_img, left_eeg, right_eeg, labels in val_loader:
                left_img, right_img, left_eeg, right_eeg, labels = (
                    left_img.to(self.device), right_img.to(self.device),
                    left_eeg.to(self.device), right_eeg.to(self.device), labels.to(self.device)
                )

                outputs = self.model(eeg1=left_eeg, eeg2=right_eeg, img1=left_img, img2=right_img, mode='train')
                loss_dict = self.model.compute_loss(outputs, labels)

                self._compute_metrics_from_outputs(outputs, labels, loss_dict, metrics_accum, arrays)

        return self._calculate_final_metrics(metrics_accum, arrays, len(val_loader))

    # ==================== 阶段循环控制器 ====================
    def train_stage(self, stage_name, train_loader, val_loader, lr_head, lr_backbone, max_epochs, patience=5,
                    min_epochs=10):
        print(f"\n{'=' * 60}")
        print(f"🌟 开始阶段: {stage_name}")
        print(f"{'=' * 60}")

        optimizer = self.build_optimizer(lr_head, lr_backbone)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=min_epochs, T_mult=1, eta_min=lr_head * 0.01
        )

        stage_best_val_loss = float('inf')
        no_improve_count = 0
        start_time = time.time()

        for epoch in range(max_epochs):
            self.global_epoch += 1
            current_temp = self._update_temperature()
            self.history['temperatures'].append(current_temp)
            current_lr = optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)

            # 训练与验证
            train_metrics = self.train_epoch(train_loader, optimizer)
            val_metrics = self.validate_epoch(val_loader)
            scheduler.step()

            # 历史记录追加
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

            # 早停与最佳模型保存判断
            if val_metrics['total_loss'] < stage_best_val_loss:
                stage_best_val_loss = val_metrics['total_loss']
                no_improve_count = 0
                self.save_checkpoint(stage_name=stage_name)
                print(f"  🌟总损失最低时的学生网络准确率: {val_metrics['student_acc']:.2f}%，模型已保存！")
            else:
                no_improve_count += 1

            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
            if val_metrics['teacher_acc'] > self.best_teacher_acc:
                self.best_teacher_acc = val_metrics['teacher_acc']
            if val_metrics['student_acc'] > self.best_student_acc:
                self.best_student_acc = val_metrics['student_acc']

            # 打印日志 (每 5 轮或第一轮)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"[{stage_name}] Epoch {epoch + 1}/{max_epochs} | 全局 Epoch: {self.global_epoch} | Temp: {current_temp:.2f}")
                print(f"  Lr: {current_lr:.2e} | 无改善: {no_improve_count}/{patience}")
                print(f"  总损失: (Train: {train_metrics['total_loss']:.4f} / Val: {val_metrics['total_loss']:.4f})")
                print(
                    f"  Teacher Acc: (Train: {train_metrics['teacher_acc']:.2f}% / Val: {val_metrics['teacher_acc']:.2f}%)")
                print(
                    f"  Student Acc: (Train: {train_metrics['student_acc']:.2f}% / Val: {val_metrics['student_acc']:.2f}%)")
                print(f"  KRR (保留率): {val_metrics['krr']:.2f}%")

            # 早停触发
            if no_improve_count >= patience and epoch >= min_epochs:
                print(f"🛑 {stage_name} 早停于本阶段第 {epoch + 1} 轮")
                break

        print(f"⏱️ {stage_name} 耗时: {(time.time() - start_time) / 60:.2f} 分钟")

    # ==================== 编排主入口 ====================
    def train(self):
        train_loader, val_loader = self.create_dataloaders()

        # 阶段1: 快速收敛 Head 与初步对齐
        self.freeze_backbones()
        self.unfreeze_heads()
        self.train_stage(
            stage_name="stage1_freeze_backbone",
            train_loader=train_loader, val_loader=val_loader,
            lr_head=1e-3, lr_backbone=0,
            max_epochs=30, patience=8, min_epochs=10
        )

        # 阶段2: 强化对齐与蒸馏收敛
        self.freeze_backbones()
        self.unfreeze_heads()
        self.train_stage(
            stage_name="stage2_align_features",
            train_loader=train_loader, val_loader=val_loader,
            lr_head=5e-4, lr_backbone=0,
            max_epochs=40, patience=5, min_epochs=10
        )

        # 阶段3: 微调 Backbone 解开封印
        dynamic_img_layers = 6 if self.config.image_model_name == 'VGG' else 4
        self.unfreeze_backbone_last_layers(eeg_layers=4, img_layers=dynamic_img_layers)
        self.unfreeze_heads()
        self.train_stage(
            stage_name="stage3_finetune_backbone",
            train_loader=train_loader, val_loader=val_loader,
            lr_head=2e-4, lr_backbone=1e-5,  # 极小的 backbone 学习率
            max_epochs=40, patience=10, min_epochs=20
        )

        print(f"\n🎉 特权学习三阶段训练全部完成!")
        print(f"最佳验证损失: {self.best_val_loss:.6f}")
        print(f"教师网络历史最高验证 Acc: {self.best_teacher_acc:.2f}%")
        print(f"学生网络历史最高验证 Acc: {self.best_student_acc:.2f}%")

        self.save_checkpoint(is_final=True)
        self._save_training_history()
        self._plot_training_results()

    # ==================== 辅助保存与绘图 ====================
    def save_checkpoint(self, stage_name=None, is_final=False):
        filename = "privileged_final_model.pth" if is_final else f"best_{stage_name}.pth"
        filepath = os.path.join(self.output_dir, filename)
        torch.save({
            'global_epoch': self.global_epoch,
            'model_state_dict': self.model.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict()
        }, filepath)

        # 如果是最后一步，同时把提纯过的学生网络单独保存出来
        if is_final:
            student_model = self.model.module.student_network if hasattr(self.model,
                                                                         'module') else self.model.student_network
            torch.save({'student_state_dict': student_model.state_dict()},
                       os.path.join(self.output_dir, "distilled_student_only.pth"))

    def _save_training_history(self):
        history_path = os.path.join(self.output_dir, "training_history.json")
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump({
                'epochs': self.global_epoch,
                'best_val_loss': self.best_val_loss,
                'best_teacher_acc': self.best_teacher_acc,
                'best_student_acc': self.best_student_acc
            }, f, indent=2, ensure_ascii=False)

    def _plot_training_results(self):
        """完全保留之前的 4 张 Seaborn 学术图表绘制逻辑"""
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        epochs = range(1, len(self.history['train_losses']) + 1)
        model_name = f"Privileged (Student: {self.model.student_modality.upper()})"

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        # 1. 总损失
        axes[0, 0].plot(epochs, self.history['train_losses'], label='Train Loss')
        axes[0, 0].plot(epochs, self.history['val_losses'], label='Val Loss')
        axes[0, 0].set_title('Total Loss', fontweight='bold')
        axes[0, 0].legend()
        # 2. 教师与学生损失
        axes[0, 1].plot(epochs, self.history['train_teacher_losses'], label='Train Teacher Loss')
        axes[0, 1].plot(epochs, self.history['val_teacher_losses'], label='Val Teacher Loss')
        axes[0, 1].plot(epochs, self.history['train_student_losses'], label='Train Student Loss')
        axes[0, 1].plot(epochs, self.history['val_student_losses'], label='Val Student Loss')
        axes[0, 1].set_title('Teacher vs Student Loss', fontweight='bold')
        axes[0, 1].legend()
        # 3. 蒸馏损失
        axes[1, 0].plot(epochs, self.history['train_distill_losses'], label='Train Distill Loss')
        axes[1, 0].plot(epochs, self.history['val_distill_losses'], label='Val Distill Loss')
        axes[1, 0].set_title('Distillation Loss', fontweight='bold')
        axes[1, 0].legend()
        # 4. 对齐差异损失
        axes[1, 1].plot(epochs, self.history['train_common_sim_losses'], label='Train Common Sim Loss')
        axes[1, 1].plot(epochs, self.history['train_private_diff_losses'], label='Train Private Diff Loss')
        axes[1, 1].set_title('Feature Alignment & Difference Loss', fontweight='bold')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'all_losses_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Acc, F1, AUC 曲线
        for metric, name in zip(['acc', 'f1', 'auc'], ['Accuracy', 'F1 Score', 'AUC-ROC']):
            plt.figure(figsize=(8, 6))
            plt.plot(epochs, self.history[f'train_teacher_{metric}'], label=f'Train Teacher {name}')
            plt.plot(epochs, self.history[f'val_teacher_{metric}'], label=f'Val Teacher {name}')
            plt.plot(epochs, self.history[f'train_student_{metric}'], label=f'Train Student {name}')
            plt.plot(epochs, self.history[f'val_student_{metric}'], label=f'Val Student {name}')
            plt.title(f'{model_name} - {name}', fontweight='bold')
            plt.xlabel('Epoch')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{metric}_curve.png'), dpi=300, bbox_inches='tight')
            plt.close()

        sns.reset_orig()
        print(f"✅ 图表保存完成！")


def main():
    config = get_config('default')
    trainer = PrivilegedTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()