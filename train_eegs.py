import torch
import torch.nn as nn
import torch.optim as optim
import config.config_eeg_model
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from datetime import datetime

from MyPP2Dataset import MyPP2Dataset, create_dataloaders
from models.eeg_models import EEGFeatureExtractor, EEGFusionNetwork, SSBCINet
from utils.early_stop import SchedulerEarlyStopper
from utils.my_transforms import transform_cnn_2


class EEGTrainer:
    def __init__(self, model, train_loader, val_loader, device='cuda',
                 lr=1e-4,
                 momentum=0.9,
                 weight_decay=1e-4,
                 patience=5,
                 factor=0.1,
                 max_plateaus=4,
                 base_model_name="EEGNet",
                 output_base_dir="outputs/outputs_eeg"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # 根据基础模型名称创建输出目录
        self.output_base_dir = output_base_dir
        self.model_output_dir = os.path.join(output_base_dir, base_model_name, "models")
        self.plot_output_dir = os.path.join(output_base_dir, base_model_name, "plots")

        # 创建目录
        os.makedirs(self.model_output_dir, exist_ok=True)
        os.makedirs(self.plot_output_dir, exist_ok=True)

        print(f"Output directories created:")
        print(f"  - Models: {self.model_output_dir}")
        print(f"  - Plots: {self.plot_output_dir}")

        # 损失函数和优化器
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=lr,momentum=momentum ,weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=patience,
                                                              factor=factor)

        # 使用SchedulerEarlyStopper
        self.early_stopper = SchedulerEarlyStopper(max_plateaus=max_plateaus)

        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        self.base_model_name = base_model_name

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (left_img, right_img, left_eeg, right_eeg, labels) in enumerate(pbar):
            # 只使用EEG数据
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
            predictions = (torch.sigmoid(logits) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{correct / total * 100:.2f}%'
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct / total * 100

        return epoch_loss, epoch_acc

    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for left_img, right_img, left_eeg, right_eeg, labels in self.val_loader:
                left_eeg = left_eeg.to(self.device)
                right_eeg = right_eeg.to(self.device)
                labels = labels.float().to(self.device)

                logits = self.model(left_eeg, right_eeg)
                loss = self.criterion(logits, labels)

                running_loss += loss.item()
                predictions = (torch.sigmoid(logits) > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = correct / total * 100

        return epoch_loss, epoch_acc

    def train(self, epochs=50):
        print(f"Starting training on {self.device}")
        print(f"Base model: {self.base_model_name}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Output base directory: {self.output_base_dir}")

        best_val_acc = 0.0
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}')
            print('-' * 50)

            # 训练
            train_loss, train_acc = self.train_epoch()

            # 验证
            val_loss, val_acc = self.validate_epoch()

            # 学习率调度
            self.scheduler.step(val_loss)

            # 检查早停条件
            should_stop = self.early_stopper.step(self.optimizer)

            # 记录
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])

            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
            print(f'Plateau count: {self.early_stopper.plateau_count}/{self.early_stopper.max_plateaus}')

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_path = os.path.join(self.model_output_dir, f'best_{self.base_model_name}_model_{timestamp}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'plateau_count': self.early_stopper.plateau_count,
                    'timestamp': timestamp,
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
                checkpoint_path = os.path.join(self.model_output_dir,
                                               f'checkpoint_{self.base_model_name}_epoch_{epoch + 1}_{timestamp}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'timestamp': timestamp,
                    'base_model_name': self.base_model_name
                }, checkpoint_path)
                print(f'Checkpoint saved at {checkpoint_path}')

            # 早停
            if should_stop:
                print(
                    f'Early stopping at epoch {epoch + 1} due to {self.early_stopper.max_plateaus} learning rate plateaus')
                break

        # 训练结束后保存最终模型
        final_model_path = os.path.join(self.model_output_dir, f'final_{self.base_model_name}_model_{timestamp}.pth')
        torch.save({
            'epoch': epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'timestamp': timestamp,
            'base_model_name': self.base_model_name,
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accuracies': self.train_accuracies,
                'val_accuracies': self.val_accuracies,
                'learning_rates': self.learning_rates
            }
        }, final_model_path)
        print(f'Final model saved at {final_model_path}')

        # 绘制训练曲线
        self.plot_training_curves(timestamp)

        return best_val_acc

    def plot_training_curves(self, timestamp):
        plt.figure(figsize=(15, 5))

        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', linewidth=2)
        plt.plot(self.val_losses, label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'{self.base_model_name} - Training and Validation Loss')
        plt.grid(True, alpha=0.45)

        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy', linewidth=2)
        plt.plot(self.val_accuracies, label='Val Accuracy', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title(f'{self.base_model_name} - Training and Validation Accuracy')
        plt.grid(True, alpha=0.45)

        # 学习率曲线
        # plt.subplot(1, 3, 3)
        # plt.plot(self.learning_rates, label='Learning Rate', color='red', linewidth=2)
        # plt.xlabel('Epoch')
        # plt.ylabel('Learning Rate')
        # plt.legend()
        # plt.title(f'{self.base_model_name} - Learning Rate Schedule')
        # plt.yscale('log')
        # plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图像
        plot_path = os.path.join(self.plot_output_dir, f'training_curves_{self.base_model_name}_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f'Training curves saved at {plot_path}')
        plt.show()

        # 保存训练数据为文本文件
        data_path = os.path.join(self.plot_output_dir, f'training_data_{self.base_model_name}_{timestamp}.txt')
        with open(data_path, 'w') as f:
            f.write(f"# Training Data for {self.base_model_name}\n")
            f.write(f"# Timestamp: {timestamp}\n")
            f.write(f"# Best Validation Accuracy: {max(self.val_accuracies):.2f}%\n")
            f.write("Epoch,Train_Loss,Val_Loss,Train_Acc,Val_Acc,Learning_Rate\n")
            for i in range(len(self.train_losses)):
                f.write(f"{i + 1},{self.train_losses[i]:.6f},{self.val_losses[i]:.6f},"
                        f"{self.train_accuracies[i]:.2f},{self.val_accuracies[i]:.2f},"
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
        subject_id="01gh"
    )

    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(
        dataset=dataset,
        train_ratio=0.8,
        batch_size=cfg.batch_size,
        shuffle=True
    )

    # 创建模型
    model = SSBCINet(
        base_model_name=cfg.base_model_name,
        in_chans=64,
        n_classes=2,
        input_window_samples=2000
    )

    print(f"Base model: {cfg.base_model_name}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 创建训练器并开始训练
    trainer = EEGTrainer(
        model,
        train_loader,
        val_loader,
        device=cfg.device,
        lr=cfg.learning_rate,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        patience=cfg.patience,
        factor=cfg.factor,
        max_plateaus=getattr(cfg, 'max_plateaus', 4),
        base_model_name=cfg.base_model_name,
        output_base_dir="outputs/outputs_eeg"
    )
    best_acc = trainer.train(epochs=cfg.num_epochs)

    print(f"\nTraining completed! Best validation accuracy: {best_acc:.2f}%")

    # 加载最佳模型进行测试
    model_dir = os.path.join("outputs/outputs_eeg", cfg.base_model_name, "models")
    model_files = [f for f in os.listdir(model_dir) if f.startswith(f"best_{cfg.base_model_name}_model_")]
    if model_files:
        latest_model = sorted(model_files)[-1]
        checkpoint_path = os.path.join(model_dir, latest_model)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Best model loaded from {checkpoint_path} for inference.")
        print(f"Best validation accuracy: {checkpoint['val_acc']:.2f}%")
        print(f"Base model: {checkpoint.get('base_model_name', 'Unknown')}")
    else:
        print("No best model found!")


if __name__ == "__main__":
    main()