import torch
import torch.nn as nn
import torch.optim as optim
import config.config_eeg_model
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns  # 新增 seaborn
import os
from datetime import datetime
from sklearn.metrics import f1_score, roc_auc_score  # 新增指标

from MyPP2Dataset import MyPP2Dataset, create_dataloaders
from models.eeg_models import EEGFeatureExtractor, EEGFusionNetwork, SSBCINet
from utils.early_stop import SchedulerEarlyStopper
from utils.my_transforms import transform_cnn_2


class EEGTrainer:
    def __init__(self, model, train_loader, val_loader, device='cuda',
                 lr=1e-4,
                 weight_decay=1e-4,
                 patience=5,
                 factor=0.1,
                 max_plateaus=4,
                 base_model_name="EEGNet",
                 subject_id="01gh",  # 新增 subject_id 参数
                 output_base_dir="outputs/outputs_eeg"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # 根据基础模型名称和被试ID创建输出目录
        self.output_base_dir = os.path.join(output_base_dir, base_model_name, subject_id)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_output_dir = os.path.join(self.output_base_dir, self.timestamp, "models")
        self.plot_output_dir = os.path.join(self.output_base_dir, self.timestamp, "plots")
        self.checkpoint_output_dir = os.path.join(self.output_base_dir, self.timestamp, "checkpoints")

        # 创建目录
        os.makedirs(self.model_output_dir, exist_ok=True)
        os.makedirs(self.plot_output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_output_dir, exist_ok=True)

        print(f"Output directories created:")
        print(f"  - Base: {self.output_base_dir}")
        print(f"  - Models: {self.model_output_dir}")
        print(f"  - Plots: {self.plot_output_dir}")
        print(f"  - Checkpoints: {self.checkpoint_output_dir}")

        # 损失函数和优化器
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=patience,
                                                              factor=factor, verbose=True)

        # 使用SchedulerEarlyStopper
        self.early_stopper = SchedulerEarlyStopper(max_plateaus=max_plateaus)

        # 训练记录 (新增了 F1 和 AUC 的列表)
        self.train_losses, self.val_losses = [], []
        self.train_accuracies, self.val_accuracies = [], []
        self.train_f1s, self.val_f1s = [], []
        self.train_aucs, self.val_aucs = [], []
        self.learning_rates = []

        self.base_model_name = base_model_name

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        all_labels = []
        all_preds = []
        all_probs = []

        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (left_img, right_img, left_eeg, right_eeg, labels) in enumerate(pbar):
            left_eeg = left_eeg.to(self.device)
            right_eeg = right_eeg.to(self.device)
            labels = labels.float().to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            logits = self.model(left_eeg, right_eeg)
            loss = self.criterion(logits, labels)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 统计
            running_loss += loss.item()
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).float()

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # 收集用于计算 F1 和 AUC 的数据
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{correct / total * 100:.2f}%'
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct / total * 100

        epoch_f1 = f1_score(all_labels, all_preds, average='macro')
        try:
            epoch_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            epoch_auc = 0.5  # 防止 batch 内只有单一类别导致报错

        return epoch_loss, epoch_acc, epoch_f1, epoch_auc

    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for left_img, right_img, left_eeg, right_eeg, labels in self.val_loader:
                left_eeg = left_eeg.to(self.device)
                right_eeg = right_eeg.to(self.device)
                labels = labels.float().to(self.device)

                logits = self.model(left_eeg, right_eeg)
                loss = self.criterion(logits, labels)

                running_loss += loss.item()
                probs = torch.sigmoid(logits)
                predictions = (probs > 0.5).float()

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predictions.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = correct / total * 100

        epoch_f1 = f1_score(all_labels, all_preds, average='macro')
        try:
            epoch_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            epoch_auc = 0.5

        return epoch_loss, epoch_acc, epoch_f1, epoch_auc

    def train(self, epochs=50):
        print(f"Starting training on {self.device}")
        print(f"Base model: {self.base_model_name}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Output base directory: {self.output_base_dir}")

        best_val_acc = 0.0

        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}')
            print('-' * 50)

            # 训练
            train_loss, train_acc, train_f1, train_auc = self.train_epoch()

            # 验证
            val_loss, val_acc, val_f1, val_auc = self.validate_epoch()

            # 学习率调度
            self.scheduler.step(val_loss)

            # 检查早停条件
            should_stop = self.early_stopper.step(self.optimizer)

            # 记录所有指标
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.train_f1s.append(train_f1)
            self.val_f1s.append(val_f1)
            self.train_aucs.append(train_auc)
            self.val_aucs.append(val_auc)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])

            print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_f1:.4f}, AUC: {train_auc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.4f}, AUC: {val_auc:.4f}')
            print(f'Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
            print(f'Plateau count: {self.early_stopper.plateau_count}/{self.early_stopper.max_plateaus}')

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_path = os.path.join(self.model_output_dir, f'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'val_f1': val_f1,
                    'val_auc': val_auc,
                    'plateau_count': self.early_stopper.plateau_count,
                    'timestamp': self.timestamp,
                    'base_model_name': self.base_model_name,
                    'config': {
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'weight_decay': self.optimizer.param_groups[0]['weight_decay'],
                        'batch_size': self.train_loader.batch_size
                    }
                }, model_path)
                print(f'Best model saved with validation accuracy: {val_acc:.2f}% at {model_path}')

            # 每10个epoch保存一次检查点
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(self.checkpoint_output_dir, f'checkpoint_epoch_{epoch + 1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'timestamp': self.timestamp,
                    'base_model_name': self.base_model_name,
                    'training_history': {
                        'train_losses': self.train_losses[:epoch + 1],
                        'val_losses': self.val_losses[:epoch + 1],
                        'train_accuracies': self.train_accuracies[:epoch + 1],
                        'val_accuracies': self.val_accuracies[:epoch + 1],
                        'train_f1s': self.train_f1s[:epoch + 1],
                        'val_f1s': self.val_f1s[:epoch + 1],
                        'train_aucs': self.train_aucs[:epoch + 1],
                        'val_aucs': self.val_aucs[:epoch + 1],
                        'learning_rates': self.learning_rates[:epoch + 1]
                    }
                }, checkpoint_path)
                print(f'Checkpoint saved at {checkpoint_path}')

            # 早停
            if should_stop:
                print(
                    f'Early stopping at epoch {epoch + 1} due to {self.early_stopper.max_plateaus} learning rate plateaus')
                break

        # 训练结束后保存最终模型
        final_model_path = os.path.join(self.model_output_dir, 'final_model.pth')
        torch.save({
            'epoch': epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'timestamp': self.timestamp,
            'base_model_name': self.base_model_name,
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accuracies': self.train_accuracies,
                'val_accuracies': self.val_accuracies,
                'train_f1s': self.train_f1s,
                'val_f1s': self.val_f1s,
                'train_aucs': self.train_aucs,
                'val_aucs': self.val_aucs,
                'learning_rates': self.learning_rates
            }
        }, final_model_path)
        print(f'Final model saved at {final_model_path}')

        # 绘制训练曲线
        self.plot_training_curves()

        return best_val_acc

    def plot_training_curves(self):
        """完全复刻图像模型的seaborn风格三图绘制方式"""

        # 设置 seaborn 学术风格
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

        epochs = range(1, len(self.train_losses) + 1)

        # ==========================
        # 图 1: Loss 和 Accuracy (双子图)
        # ==========================
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss 子图
        ax1.plot(epochs, self.train_losses, linestyle='-', marker='o', markersize=4, label='Train Loss', linewidth=2,
                 alpha=0.8)
        ax1.plot(epochs, self.val_losses, linestyle='-', marker='s', markersize=4, label='Val Loss', linewidth=2,
                 alpha=0.8)
        ax1.set_title(f'{self.base_model_name} - Loss', fontweight='bold')
        ax1.set_xlabel('Epoch', fontweight='bold')
        ax1.set_ylabel('Loss', fontweight='bold')
        ax1.legend(loc='upper right', frameon=True, shadow=True)

        # Accuracy 子图
        ax2.plot(epochs, self.train_accuracies, linestyle='-', marker='o', markersize=4, label='Train Acc', linewidth=2,
                 alpha=0.8)
        ax2.plot(epochs, self.val_accuracies, linestyle='-', marker='s', markersize=4, label='Val Acc', linewidth=2,
                 alpha=0.8)
        ax2.set_title(f'{self.base_model_name} - Accuracy', fontweight='bold')
        ax2.set_xlabel('Epoch', fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontweight='bold')
        ax2.legend(loc='lower right', frameon=True, shadow=True)

        plt.tight_layout()
        loss_acc_path = os.path.join(self.plot_output_dir, "loss_acc_curves.png")
        plt.savefig(loss_acc_path, dpi=300, bbox_inches='tight')
        plt.close()

        # ==========================
        # 图 2: F1 分数 (单独的图)
        # ==========================
        plt.figure(figsize=(7, 5))
        plt.plot(epochs, self.train_f1s, linestyle='-', marker='o', markersize=4, label='Train F1', linewidth=2,
                 alpha=0.8)
        plt.plot(epochs, self.val_f1s, linestyle='-', marker='s', markersize=4, label='Val F1', linewidth=2, alpha=0.8)
        plt.title(f'{self.base_model_name} - F1 Score', fontweight='bold')
        plt.xlabel('Epoch', fontweight='bold')
        plt.ylabel('F1 Score', fontweight='bold')
        plt.legend(loc='lower right', frameon=True, shadow=True)

        plt.tight_layout()
        f1_path = os.path.join(self.plot_output_dir, "f1_curve.png")
        plt.savefig(f1_path, dpi=300, bbox_inches='tight')
        plt.close()

        # ==========================
        # 图 3: AUC-ROC (单独的图)
        # ==========================
        plt.figure(figsize=(7, 5))
        plt.plot(epochs, self.train_aucs, linestyle='-', marker='o', markersize=4, label='Train AUC', linewidth=2,
                 alpha=0.8)
        plt.plot(epochs, self.val_aucs, linestyle='-', marker='s', markersize=4, label='Val AUC', linewidth=2,
                 alpha=0.8)
        plt.title(f'{self.base_model_name} - AUC-ROC', fontweight='bold')
        plt.xlabel('Epoch', fontweight='bold')
        plt.ylabel('AUC-ROC', fontweight='bold')
        plt.legend(loc='lower right', frameon=True, shadow=True)

        plt.tight_layout()
        auc_path = os.path.join(self.plot_output_dir, "auc_roc_curve.png")
        plt.savefig(auc_path, dpi=300, bbox_inches='tight')
        plt.close()

        sns.reset_orig()
        print(f"✅ 三个绘图已保存:\n  1. {loss_acc_path}\n  2. {f1_path}\n  3. {auc_path}")

        # 保存包含所有指标的数据为文本文件
        data_path = os.path.join(self.plot_output_dir, 'training_data.txt')
        with open(data_path, 'w') as f:
            f.write(f"# Training Data for {self.base_model_name}\n")
            f.write(f"# Timestamp: {self.timestamp}\n")
            f.write(f"# Best Validation Accuracy: {max(self.val_accuracies):.2f}%\n")
            f.write("Epoch,Train_Loss,Val_Loss,Train_Acc,Val_Acc,Train_F1,Val_F1,Train_AUC,Val_AUC,Learning_Rate\n")
            for i in range(len(self.train_losses)):
                f.write(f"{i + 1},{self.train_losses[i]:.6f},{self.val_losses[i]:.6f},"
                        f"{self.train_accuracies[i]:.2f},{self.val_accuracies[i]:.2f},"
                        f"{self.train_f1s[i]:.4f},{self.val_f1s[i]:.4f},"
                        f"{self.train_aucs[i]:.4f},{self.val_aucs[i]:.4f},"
                        f"{self.learning_rates[i]:.8f}\n")
        print(f'Training data saved at {data_path}')


# 主训练函数
def main():
    cfg = config.config_eeg_model.Config()
    # 设置设备
    device = torch.device(cfg.device)
    print(f'Using device: {device}')

    # 创建数据集
    dataset = MyPP2Dataset(
        csv_file="data/safe_qscores_high2low.xlsx",
        transform=transform_cnn_2,
        img_dir="data",
        eeg_dir="data/EEG/seg_eeg_data",
        is_flipped=False,
        subject_id=cfg.subject_id
    )

    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(
        dataset=dataset,
        train_ratio=0.8,
        batch_size=cfg.batch_size,
        shuffle=True
    )

    # 检查数据分布
    train_labels = [label for _, _, _, _, label in train_loader.dataset]
    test_labels = [label for _, _, _, _, label in val_loader.dataset]
    print("==========================检查数据集分布==========================")
    print(f"训练集分布: {torch.bincount(torch.tensor(train_labels)).tolist()}")
    print(f"测试集分布: {torch.bincount(torch.tensor(test_labels)).tolist()}")
    print(f"训练集: {len(train_loader.dataset)}, 测试集: {len(val_loader.dataset)}")

    # 创建模型
    model = SSBCINet(
        base_model_name=cfg.base_model_name,
        in_chans=64,
        n_classes=2,
        input_window_samples=2000
    )

    print(f"Base model: {cfg.base_model_name}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 创建训练器并开始训练（新增传入了 cfg.subject_id）
    trainer = EEGTrainer(
        model,
        train_loader,
        val_loader,
        device=cfg.device,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        patience=cfg.patience,
        factor=cfg.factor,
        max_plateaus=cfg.max_lr_plateaus,
        base_model_name=cfg.base_model_name,
        subject_id=cfg.subject_id,  # 将配置的被试ID传给训练器以便创建独立文件夹
        output_base_dir="outputs/outputs_eeg"
    )
    best_acc = trainer.train(epochs=cfg.num_epochs)

    print(f"\nTraining completed! Best validation accuracy: {best_acc:.2f}%")

    # 从正确的路径加载最佳模型
    best_model_path = os.path.join(trainer.model_output_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Best model loaded from {best_model_path} for inference.")
        print(f"Best validation accuracy: {checkpoint['val_acc']:.2f}%")
        print(f"Base model: {checkpoint.get('base_model_name', 'Unknown')}")
        print(f"Training timestamp: {checkpoint.get('timestamp', 'Unknown')}")
    else:
        print(f"No best model found at {best_model_path}!")


if __name__ == "__main__":
    main()