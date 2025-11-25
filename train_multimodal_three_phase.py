import torch
import torch.optim as optim
import time
from datetime import datetime
import matplotlib.pyplot as plt
from MyPP2Dataset import MyPP2Dataset, create_dataloaders
from models.multi_model import MultiModalFusionNetwork
from config import config_multi_model
import os
from tqdm import tqdm


class MultiStageTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.history = {
            'stage1': [], 'stage2': [], 'stage3': [],
            'all_train_loss': [], 'all_val_loss': [],
            'all_train_acc': [], 'all_val_acc': [],
            'all_task_loss': [], 'all_common_sim_loss': [], 'all_private_diff_loss': []
        }

    def setup_data(self):
        """数据准备"""
        dataset = MyPP2Dataset(
            is_flipped=False,
            transform=self.config.transform,
            subject_id=self.config.subject_id
        )
        self.train_loader, self.test_loader = create_dataloaders(
            dataset=dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        # 检查数据分布
        train_labels = [label for _, _, _, _, label in self.train_loader.dataset]
        test_labels = [label for _, _, _, _, label in self.test_loader.dataset]

        print(f"训练集分布: {torch.bincount(torch.tensor(train_labels)).tolist()}")
        print(f"测试集分布: {torch.bincount(torch.tensor(test_labels)).tolist()}")
        print(f"训练集: {len(self.train_loader.dataset)}, 测试集: {len(self.test_loader.dataset)}")

    def setup_model(self):
        """模型初始化"""
        self.model = MultiModalFusionNetwork(
            eeg_model_name=self.config.base_eeg_model,
            image_model_name=self.config.base_image_model,
            image_model_type=self.config.image_model_type,
            in_chans=64,
            n_classes=2,
            input_window_samples=2000,
            freeze_eeg_backbone=False,
            freeze_image_backbone=False,
            common_dim=512,
            private_dim=256,
            dropout_rate=0.5,
            alpha=self.config.alpha,
            beta=self.config.beta,
        ).to(self.device)

        # 参数统计
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"总参数: {total_params:,}, 可训练参数: {trainable_params:,}")

    def freeze_backbones(self):
        """冻结backbone"""
        for p in self.model.eeg_feature_net.parameters():
            p.requires_grad = False
        for p in self.model.image_feature_net.parameters():
            p.requires_grad = False
        print("🔒 冻结所有backbone参数")

    def unfreeze_backbone_last_layers(self, eeg_layers=4, img_layers=4):
        """解冻backbone最后几层 - 优化版本"""

        # EEG解冻策略
        eeg_modules = list(self.model.eeg_feature_net.feature_extractor.feature_net.children())

        # 解冻EEG最后几层（从后往前）
        eeg_unfreeze_indices = list(range(len(eeg_modules)))[-eeg_layers:]
        for idx in eeg_unfreeze_indices:
            for p in eeg_modules[idx].parameters():
                p.requires_grad = True
        print(f"🔓 解冻EEG层: {eeg_unfreeze_indices} - {[type(m).__name__ for m in eeg_modules[-eeg_layers:]]}")

        # Image解冻策略
        img_modules = list(self.model.image_feature_net.feature_extractor.features.children())

        # 解冻Image最后几层（从后往前）
        img_unfreeze_indices = list(range(len(img_modules)))[-img_layers:]
        for idx in img_unfreeze_indices:
            for p in img_modules[idx].parameters():
                p.requires_grad = True
        print(f"🔓 解冻Image层: {img_unfreeze_indices} - {[type(m).__name__ for m in img_modules[-img_layers:]]}")

        # 额外解冻融合层（重要！）
        for p in self.model.eeg_feature_net.fusion.parameters():
            p.requires_grad = True
        for p in self.model.image_feature_net.fusion_convs.parameters():
            p.requires_grad = True
        print("🔓 额外解冻融合层")

    def unfreeze_heads(self):
        """解冻head网络"""
        for p in self.model.common_encoder.parameters():
            p.requires_grad = True
        for p in self.model.eeg_private_encoder.parameters():
            p.requires_grad = True
        for p in self.model.image_private_encoder.parameters():
            p.requires_grad = True
        for p in self.model.classifier.parameters():
            p.requires_grad = True
        print("🔓 解冻head网络")

    def build_optimizer(self, lr_head, lr_backbone):
        """构建分组优化器"""
        head_params = []
        backbone_params = []

        for name, p in self.model.named_parameters():
            if p.requires_grad:
                if "feature_net" in name:
                    backbone_params.append(p)
                else:
                    head_params.append(p)

        param_groups = []
        if head_params:
            param_groups.append({'params': head_params, 'lr': lr_head})
        if backbone_params:
            param_groups.append({'params': backbone_params, 'lr': lr_backbone})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.config.weight_decay)

        print(f"🔧 优化器: head_lr={lr_head}, backbone_lr={lr_backbone}")
        print(f"   可训练参数: head={len(head_params)}组, backbone={len(backbone_params)}组")

        return optimizer

    def train_stage(self, stage_name, lr_head, lr_backbone,
                    max_epochs, patience=5, min_epochs=10):
        """
        智能训练阶段
        Args:
            patience: 验证损失不改善的容忍轮次
            min_epochs: 最小训练轮次
        """
        print(f"\n{'=' * 50}")
        print(f"阶段: {stage_name}")
        print(f"{'=' * 50}")

        # 优化器
        optimizer = self.build_optimizer(lr_head, lr_backbone)

        # 学习率调度 - 使用CosineAnnealingWarmRestarts更平滑
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=min_epochs, T_mult=1, eta_min=lr_head * 0.01
        )

        best_val_loss = float('inf')
        no_improve_count = 0
        stage_history = []

        start_time = time.time()

        for epoch in range(max_epochs):
            # 训练
            train_results = self.train_epoch(optimizer, epoch, max_epochs, stage_name)
            train_loss, train_acc, task_loss, common_sim_loss, private_diff_loss = train_results

            # 验证
            val_results = self.validate_epoch(epoch, max_epochs, stage_name)
            val_loss, val_acc, val_task_loss, val_common_sim_loss, val_private_diff_loss = val_results

            # 学习率调度
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            # 记录历史
            stage_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss, 'val_loss': val_loss,
                'train_acc': train_acc, 'val_acc': val_acc,
                'lr': current_lr,
                'task_loss': task_loss, 'common_sim_loss': common_sim_loss,
                'private_diff_loss': private_diff_loss,
                'val_task_loss': val_task_loss, 'val_common_sim_loss': val_common_sim_loss,
                'val_private_diff_loss': val_private_diff_loss
            })

            self.history['all_train_loss'].append(train_loss)
            self.history['all_val_loss'].append(val_loss)
            self.history['all_train_acc'].append(train_acc)
            self.history['all_val_acc'].append(val_acc)
            self.history['all_task_loss'].append(task_loss)
            self.history['all_common_sim_loss'].append(common_sim_loss)
            self.history['all_private_diff_loss'].append(private_diff_loss)

            # 早停判断
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_count = 0
                # 保存最佳模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'stage': stage_name
                }, os.path.join(self.save_dir, f'best_{stage_name}.pth'))
                print(f"  ✅ 保存最佳模型 (val_loss: {val_loss:.4f})")
            else:
                no_improve_count += 1

            # 打印进度
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"\nEpoch {epoch + 1}/{max_epochs} 总结:")
                print(f"  训练 - 总损失: {train_loss:.4f}, 任务损失: {task_loss:.4f}, "
                      f"共同特征相似度损失: {common_sim_loss:.4f}, 私有特征差异损失: {private_diff_loss:.4f}")
                print(f"  训练准确率: {train_acc:.2f}%")
                print(f"  验证 - 总损失: {val_loss:.4f}, 任务损失: {val_task_loss:.4f}, "
                      f"共同特征相似度损失: {val_common_sim_loss:.4f}, 私有特征差异损失: {val_private_diff_loss:.4f}")
                print(f"  验证准确率: {val_acc:.2f}%")
                print(f"  学习率: {current_lr:.2e}, 无改善轮次: {no_improve_count}/{patience}")

            # 早停条件
            if no_improve_count >= patience and epoch >= min_epochs:
                print(f"🛑 {stage_name} 早停于第 {epoch + 1} 轮")
                break

        training_time = time.time() - start_time
        print(f"⏱️  {stage_name} 训练时间: {training_time / 60:.1f}分钟")

        self.history[stage_name] = stage_history
        return stage_history

    def train_epoch(self, optimizer, epoch, max_epochs, stage_name):
        """训练一个epoch"""
        self.model.train()
        train_loss = 0.0
        train_task_loss = 0.0
        train_common_sim_loss = 0.0
        train_private_diff_loss = 0.0
        train_correct = 0
        train_total = 0

        # 创建tqdm进度条
        pbar = tqdm(self.train_loader, desc=f'{stage_name} Epoch {epoch + 1}/{max_epochs} [训练]')

        for batch_idx, (left_img, right_img, left_eeg, right_eeg, labels) in enumerate(pbar):
            # 数据迁移
            left_img = left_img.to(self.device)
            right_img = right_img.to(self.device)
            left_eeg = left_eeg.to(self.device)
            right_eeg = right_eeg.to(self.device)
            labels = labels.to(self.device)

            # 前向传播
            optimizer.zero_grad()
            logits, eeg_common, image_common, eeg_private, image_private = self.model(
                left_eeg, right_eeg, left_img, right_img
            )

            # 损失计算
            losses = self.model.compute_loss(
                eeg_common, image_common, eeg_private, image_private, logits, labels
            )

            total_loss = losses['total_loss']
            task_loss = losses['task_loss']
            common_sim_loss = losses['common_sim_loss']
            private_diff_loss = losses['private_diff_loss']

            # 反向传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()

            # 统计
            train_loss += total_loss.item()
            train_task_loss += task_loss.item()
            train_common_sim_loss += common_sim_loss.item()
            train_private_diff_loss += private_diff_loss.item()

            preds = (logits.squeeze() > 0).long() if logits.dim() == 1 else torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            # 更新进度条
            pbar.set_postfix({
                '总损失': f'{total_loss.item():.4f}',
                '任务损失': f'{task_loss.item():.4f}',
                '共同损失': f'{common_sim_loss.item():.4f}',
                '私有损失': f'{private_diff_loss.item():.4f}',
                '准确率': f'{100.0 * train_correct / train_total:.2f}%'
            })

        # 计算epoch平均值
        avg_train_loss = train_loss / len(self.train_loader)
        avg_task_loss = train_task_loss / len(self.train_loader)
        avg_common_sim_loss = train_common_sim_loss / len(self.train_loader)
        avg_private_diff_loss = train_private_diff_loss / len(self.train_loader)
        train_accuracy = 100.0 * train_correct / train_total

        return avg_train_loss, train_accuracy, avg_task_loss, avg_common_sim_loss, avg_private_diff_loss

    def validate_epoch(self, epoch, max_epochs, stage_name):
        """验证一个epoch"""
        self.model.eval()
        val_loss = 0.0
        val_task_loss = 0.0
        val_common_sim_loss = 0.0
        val_private_diff_loss = 0.0
        val_correct = 0
        val_total = 0

        # 创建tqdm进度条
        pbar = tqdm(self.test_loader, desc=f'{stage_name} Epoch {epoch + 1}/{max_epochs} [验证]')

        with torch.no_grad():
            for batch_idx, (left_img, right_img, left_eeg, right_eeg, labels) in enumerate(pbar):
                # 数据迁移
                left_img = left_img.to(self.device)
                right_img = right_img.to(self.device)
                left_eeg = left_eeg.to(self.device)
                right_eeg = right_eeg.to(self.device)
                labels = labels.to(self.device)

                # 前向传播
                logits, eeg_common, image_common, eeg_private, image_private = self.model(
                    left_eeg, right_eeg, left_img, right_img
                )

                # 损失计算
                losses = self.model.compute_loss(
                    eeg_common, image_common, eeg_private, image_private, logits, labels
                )

                total_loss = losses['total_loss']
                task_loss = losses['task_loss']
                common_sim_loss = losses['common_sim_loss']
                private_diff_loss = losses['private_diff_loss']

                val_loss += total_loss.item()
                val_task_loss += task_loss.item()
                val_common_sim_loss += common_sim_loss.item()
                val_private_diff_loss += private_diff_loss.item()

                preds = (logits.squeeze() > 0).long() if logits.dim() == 1 else torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                # 更新进度条
                pbar.set_postfix({
                    '总损失': f'{total_loss.item():.4f}',
                    '任务损失': f'{task_loss.item():.4f}',
                    '共同损失': f'{common_sim_loss.item():.4f}',
                    '私有损失': f'{private_diff_loss.item():.4f}',
                    '准确率': f'{100.0 * val_correct / val_total:.2f}%'
                })

        # 计算epoch平均值
        avg_val_loss = val_loss / len(self.test_loader)
        avg_val_task_loss = val_task_loss / len(self.test_loader)
        avg_val_common_sim_loss = val_common_sim_loss / len(self.test_loader)
        avg_val_private_diff_loss = val_private_diff_loss / len(self.test_loader)
        val_accuracy = 100.0 * val_correct / val_total

        return avg_val_loss, val_accuracy, avg_val_task_loss, avg_val_common_sim_loss, avg_val_private_diff_loss

    def train(self):
        """主训练流程"""
        # 设置保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = f"outputs/outputs_multi_model/{self.config.base_eeg_model}+{self.config.base_image_model}_{self.config.image_model_type}/multimodal_{timestamp}"
        os.makedirs(self.save_dir, exist_ok=True)

        # 阶段1: 快速收敛head
        self.freeze_backbones()
        self.unfreeze_heads()
        self.train_stage(
            stage_name="stage1_freeze_backbone",
            lr_head=1e-3,
            lr_backbone=0,
            max_epochs=30,
            patience=8,
            min_epochs=10
        )

        # 阶段2: 强化对齐
        self.freeze_backbones()
        self.unfreeze_heads()
        self.train_stage(
            stage_name="stage2_align_features",
            lr_head=5e-4,
            lr_backbone=0,
            max_epochs=80,
            patience=10,
            min_epochs=30
        )

        # 阶段3: 微调backbone
        # 非VGG解冻前四层即可
        self.unfreeze_backbone_last_layers(eeg_layers=8, img_layers=8)
        self.unfreeze_heads()
        self.train_stage(
            stage_name="stage3_finetune_backbone",
            lr_head=2e-4,
            lr_backbone=1e-5,
            max_epochs=150,
            patience=10,
            min_epochs=100
        )

        # 保存最终模型和训练历史
        self.save_training_history()
        print(f"\n🎉 训练完成! 所有文件保存在: {self.save_dir}")

        return self.model, self.history

    def save_training_history(self):
        """保存训练历史"""
        # 保存历史数据
        torch.save(self.history, os.path.join(self.save_dir, 'training_history.pth'))

        # 绘制训练曲线
        self.plot_training_curves()

    def plot_training_curves(self):
        """绘制训练曲线"""
        plt.figure(figsize=(20, 15))

        epochs = range(1, len(self.history['all_train_loss']) + 1)

        # 总损失曲线
        plt.subplot(2, 3, 1)
        plt.plot(epochs, self.history['all_train_loss'], label='Train Loss', linewidth=2)
        plt.plot(epochs, self.history['all_val_loss'], label='Val Loss', linewidth=2)
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 准确率曲线
        plt.subplot(2, 3, 2)
        plt.plot(epochs, self.history['all_train_acc'], label='Train Acc', linewidth=2)
        plt.plot(epochs, self.history['all_val_acc'], label='Val Acc', linewidth=2)
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 任务损失曲线
        plt.subplot(2, 3, 3)
        plt.plot(epochs, self.history['all_task_loss'], label='Task Loss', linewidth=2, color='red')
        plt.title('Task Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 共同特征相似度损失曲线
        plt.subplot(2, 3, 4)
        plt.plot(epochs, self.history['all_common_sim_loss'], label='Common Sim Loss', linewidth=2, color='green')
        plt.title('Common Similarity Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 私有特征差异损失曲线
        plt.subplot(2, 3, 5)
        plt.plot(epochs, self.history['all_private_diff_loss'], label='Private Diff Loss', linewidth=2, color='purple')
        plt.title('Private Difference Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 所有损失曲线对比
        plt.subplot(2, 3, 6)
        plt.plot(epochs, self.history['all_task_loss'], label='Task Loss', linewidth=1)
        plt.plot(epochs, self.history['all_common_sim_loss'], label='Common Sim Loss', linewidth=1)
        plt.plot(epochs, self.history['all_private_diff_loss'], label='Private Diff Loss', linewidth=1)
        plt.title('All Loss Components')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()


def train_multimodal():
    """主训练函数"""
    config = config_multi_model.Config()
    print(f"使用设备: {config.device}")

    trainer = MultiStageTrainer(config)
    trainer.setup_data()
    trainer.setup_model()

    model, history = trainer.train()
    return model, history


if __name__ == "__main__":
    model, history = train_multimodal()