import os
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import json
import pandas as pd
from datetime import datetime

import config.config_image_model
from MyPP2Dataset import MyPP2Dataset, create_dataloaders, create_subject_dataloaders
from models.image_models import SSCNN, RSSCNN
from utils.save_confusion_matrix import save_confusion_matrix
from tqdm import tqdm

from utils.early_stop import SchedulerEarlyStopper


class ImageModelTrainer:
    def __init__(self, model_type, base_model_name, timestamp):
        """
        图像模型训练器基类

        Args:
            model_type: 模型类型 ('sscnn' 或 'rsscnn')
            base_model_name: 基础模型名称
            timestamp: 时间戳
        """
        self.model_type = model_type
        self.base_model_name = base_model_name
        self.timestamp = timestamp

        # 创建输出目录结构
        self.setup_directories()

    def setup_directories(self):
        """创建输出目录结构"""
        # 基础目录
        self.base_dir = f"outputs/outputs_images/{self.model_type}/{self.base_model_name}/{self.timestamp}"

        # 子目录
        self.model_dir = os.path.join(self.base_dir, "models")
        self.plot_dir = os.path.join(self.base_dir, "plots")
        self.confusion_matrix_dir = os.path.join(self.base_dir, "confusion_matrices")
        self.data_dir = os.path.join(self.base_dir, "training_data")
        self.checkpoints_dir = os.path.join(self.base_dir, "checkpoints")  # 新增检查点目录

        # 创建所有目录
        for directory in [self.model_dir, self.plot_dir, self.confusion_matrix_dir,
                         self.data_dir, self.checkpoints_dir]:
            os.makedirs(directory, exist_ok=True)

        print(f"输出目录结构:")
        print(f"  - 基础目录: {self.base_dir}")
        print(f"  - 模型目录: {self.model_dir}")
        print(f"  - 检查点目录: {self.checkpoints_dir}")
        print(f"  - 图像目录: {self.plot_dir}")
        print(f"  - 混淆矩阵: {self.confusion_matrix_dir}")
        print(f"  - 训练数据: {self.data_dir}")


# 训练SSCNN模型
def train_sscnn(model_name='AlexNet', num_epochs=300, lr=0.001, batch_size=4, early_stopper=None,
                momentum=0.9,
                weight_decay=1e-4,
                factor=0.1,
                patience=10,
                device=None,
                my_dataset=None):
    # 创建训练器实例
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trainer = ImageModelTrainer("sscnn", model_name, timestamp)

    train_loader, val_loader = create_dataloaders(my_dataset, batch_size=batch_size, shuffle=True)

    # 检查数据分布
    train_labels = [label for _, _, _, _, label in train_loader.dataset]
    test_labels = [label for _, _, _, _, label in val_loader.dataset]
    print("==========================检查数据集分布==========================")
    print(f"训练集分布: {torch.bincount(torch.tensor(train_labels)).tolist()}")
    print(f"测试集分布: {torch.bincount(torch.tensor(test_labels)).tolist()}")
    print(f"训练集: {len(train_loader.dataset)}, 测试集: {len(val_loader.dataset)}")

    model = SSCNN(base_model_name=model_name).to(device)
    model.device = device
    criterion = nn.CrossEntropyLoss()
    # AlexNet PlacesNet
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    # VGG
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=factor, patience=patience)

    # 扩展历史记录以包含更多信息
    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'learning_rates': [], 'epoch_times': [], 'best_epoch': 0,
        'checkpoint_epochs': []  # 记录保存检查点的epoch
    }

    # 训练配置信息
    train_config = {
        'model_type': 'SSCNN',
        'base_model_name': model_name,
        'num_epochs': num_epochs,
        'learning_rate': lr,
        'batch_size': batch_size,
        'momentum': momentum,
        'factor': factor,
        'patience': patience,
        'timestamp': timestamp,
        'device': str(device)
    }

    best_acc = 0.0
    current_lr = lr

    for epoch in range(num_epochs):
        epoch_start_time = datetime.now()

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for left_imgs, right_imgs, _, _, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            left_imgs = left_imgs.to(device)
            right_imgs = right_imgs.to(device)
            labels = labels.to(device)

            outputs = model(left_imgs, right_imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        val_loss, val_acc, val_labels, val_preds = evaluate_sscnn(model, val_loader, criterion, return_preds=True)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)
        history['epoch_times'].append((datetime.now() - epoch_start_time).total_seconds())

        # 学习率更新
        old_lr = current_lr
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != old_lr:
            print(f"学习率更新: {old_lr:.6f} -> {current_lr:.6f}")

        # 保存混淆矩阵
        save_confusion_matrix(val_labels, val_preds,
                              class_names=['Left', 'Right'],
                              title=f"SSCNN {model_name} - Epoch {epoch + 1}",
                              filename=os.path.join(trainer.confusion_matrix_dir, f"confmat_epoch{epoch + 1}.png")
                              )

        # 1. 保存验证集最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            history['best_epoch'] = epoch + 1

            model_path = os.path.join(trainer.model_dir, f'best_val_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'config': train_config,
                'model_type': 'best_val'  # 标记为最佳验证模型
            }, model_path)

            # 保存最佳混淆矩阵
            save_confusion_matrix(val_labels, val_preds,
                                  class_names=['Left', 'Right'],
                                  title=f"SSCNN {model_name} - Best Val Epoch {epoch + 1}",
                                  filename=os.path.join(trainer.confusion_matrix_dir, "confmat_best_val.png")
                                  )
            print(f"✅ 保存最佳验证模型 - 准确率: {val_acc:.2f}%")

        # 2. 每10个epoch保存检查点
        if (epoch + 1) % 10 == 0 or epoch == 0:  # 第一个epoch也保存
            checkpoint_path = os.path.join(trainer.checkpoints_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'learning_rate': current_lr,
                'config': train_config,
                'model_type': 'checkpoint'
            }, checkpoint_path)
            history['checkpoint_epochs'].append(epoch + 1)
            print(f"📁 保存检查点 - Epoch {epoch + 1}")

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # 早停判断
        if early_stopper.step(optimizer):
            print("训练提前终止：学习率已经衰减达到最大次数。")
            break

    # 3. 训练完成后保存最终模型
    final_model_path = os.path.join(trainer.model_dir, "final_model.pth")
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'config': train_config,
        'history': history,
        'model_type': 'final'
    }, final_model_path)
    print(f"🏁 保存最终模型 - Epoch {num_epochs}")

    # 保存所有训练结果
    save_training_results(history, train_config, trainer, model, "SSCNN")

    return model, history, trainer.base_dir


# 训练RSSCNN模型
def train_rsscnn(model_name='AlexNet', num_epochs=300, lr=0.001, lambda_r=0.1, batch_size=4,
                 early_stopper=None,
                 momentum=0.9,
                 weight_decay=1e-4,
                 factor=0.1,
                 patience=10,
                 device=None, my_dataset_1=None):
    # 创建训练器实例
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trainer = ImageModelTrainer("rsscnn", model_name, timestamp)

    train_loader, val_loader = create_dataloaders(my_dataset_1, batch_size=batch_size, shuffle=True)

    # 检查数据分布
    train_labels = [label for _, _, _, _, label in train_loader.dataset]
    test_labels = [label for _, _, _, _, label in val_loader.dataset]
    print("==========================检查数据集分布==========================")
    print(f"训练集分布: {torch.bincount(torch.tensor(train_labels)).tolist()}")
    print(f"测试集分布: {torch.bincount(torch.tensor(test_labels)).tolist()}")
    print(f"训练集: {len(train_loader.dataset)}, 测试集: {len(val_loader.dataset)}")

    model = RSSCNN(base_model_name=model_name, lambda_r=lambda_r).to(device)
    model.device = device
    # AlexNet placesNet
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=factor, patience=patience)

    # 扩展历史记录
    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'learning_rates': [], 'epoch_times': [], 'best_epoch': 0,
        'checkpoint_epochs': []
    }

    # 训练配置信息
    train_config = {
        'model_type': 'RSSCNN',
        'base_model_name': model_name,
        'num_epochs': num_epochs,
        'learning_rate': lr,
        'lambda_r': lambda_r,
        'batch_size': batch_size,
        'momentum': momentum,
        'factor': factor,
        'patience': patience,
        'timestamp': timestamp,
        'device': str(device)
    }

    best_acc = 0.0
    current_lr = lr

    for epoch in range(num_epochs):
        epoch_start_time = datetime.now()

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for left_imgs, right_imgs, _, _, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            left_imgs, right_imgs, labels = left_imgs.to(device), right_imgs.to(device), labels.to(device)

            class_out, rank1, rank2 = model(left_imgs, right_imgs)
            loss = model.compute_loss(class_out, rank1, rank2, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(class_out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        val_loss, val_acc, val_labels, val_preds = evaluate_rsscnn(model, val_loader, return_preds=True)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)
        history['epoch_times'].append((datetime.now() - epoch_start_time).total_seconds())

        # 学习率更新
        old_lr = current_lr
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != old_lr:
            print(f"学习率更新: {old_lr:.6f} -> {current_lr:.6f}")

        # 保存混淆矩阵
        save_confusion_matrix(val_labels, val_preds,
                              class_names=['Left', 'Right'],
                              title=f"RSSCNN {model_name} - Epoch {epoch + 1}",
                              filename=os.path.join(trainer.confusion_matrix_dir, f"confmat_epoch{epoch + 1}.png")
                              )

        # 1. 保存验证集最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            history['best_epoch'] = epoch + 1

            model_path = os.path.join(trainer.model_dir, f'best_val_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'config': train_config,
                'model_type': 'best_val'
            }, model_path)

            # 保存最佳混淆矩阵
            save_confusion_matrix(val_labels, val_preds,
                                  class_names=['Left', 'Right'],
                                  title=f"RSSCNN {model_name} - Best Val Epoch {epoch + 1}",
                                  filename=os.path.join(trainer.confusion_matrix_dir, "confmat_best_val.png")
                                  )
            print(f"✅ 保存最佳验证模型 - 准确率: {val_acc:.2f}%")

        # 2. 每10个epoch保存检查点
        if (epoch + 1) % 10 == 0 or epoch == 0:  # 第一个epoch也保存
            checkpoint_path = os.path.join(trainer.checkpoints_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'learning_rate': current_lr,
                'config': train_config,
                'model_type': 'checkpoint'
            }, checkpoint_path)
            history['checkpoint_epochs'].append(epoch + 1)
            print(f"📁 保存检查点 - Epoch {epoch + 1}")

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # 早停判断
        if early_stopper.step(optimizer):
            print("训练提前终止：学习率已经衰减达到最大次数。")
            break

    # 3. 训练完成后保存最终模型
    final_model_path = os.path.join(trainer.model_dir, "final_model.pth")
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'config': train_config,
        'history': history,
        'model_type': 'final'
    }, final_model_path)
    print(f"🏁 保存最终模型 - Epoch {num_epochs}")

    # 保存所有训练结果
    save_training_results(history, train_config, trainer, model, "RSSCNN")

    return model, history, trainer.base_dir


def save_training_results(history, train_config, trainer, model, model_name):
    """保存训练结果，包括图表和数据"""

    # 1. 保存训练配置
    config_path = os.path.join(trainer.data_dir, "training_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(train_config, f, indent=2, ensure_ascii=False)

    # 2. 保存训练历史数据为CSV
    df_data = {
        'epoch': list(range(1, len(history['train_loss']) + 1)),
        'train_loss': history['train_loss'],
        'train_acc': history['train_acc'],
        'val_loss': history['val_loss'],
        'val_acc': history['val_acc'],
        'learning_rate': history['learning_rates'],
        'epoch_time_seconds': history['epoch_times']
    }

    df = pd.DataFrame(df_data)
    csv_path = os.path.join(trainer.data_dir, "training_history.csv")
    df.to_csv(csv_path, index=False)

    # 3. 保存训练摘要
    summary = {
        'best_epoch': history['best_epoch'],
        'best_val_accuracy': max(history['val_acc']) if history['val_acc'] else 0,
        'best_train_accuracy': max(history['train_acc']) if history['train_acc'] else 0,
        'final_val_accuracy': history['val_acc'][-1] if history['val_acc'] else 0,
        'final_train_accuracy': history['train_acc'][-1] if history['train_acc'] else 0,
        'total_training_time_seconds': sum(history['epoch_times']),
        'total_epochs': len(history['train_loss']),
        'final_learning_rate': history['learning_rates'][-1] if history['learning_rates'] else 0,
        'checkpoint_epochs': history['checkpoint_epochs']
    }

    summary_path = os.path.join(trainer.data_dir, "training_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 4. 绘制并保存训练曲线
    plot_training_curves(history, model_name, train_config['base_model_name'], trainer.plot_dir)

    print(f"\n训练结果已保存到: {trainer.base_dir}")
    print(f"最佳验证准确率: {summary['best_val_accuracy']:.2f}% (第 {summary['best_epoch']} 轮)")
    print(f"检查点保存轮次: {summary['checkpoint_epochs']}")
    print(f"总训练时间: {summary['total_training_time_seconds']:.2f} 秒")


def plot_training_curves(history, model_name, base_model_name, plot_dir):
    """绘制训练曲线"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    epochs = range(1, len(history['train_loss']) + 1)

    # 损失曲线
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_title(f'{model_name} {base_model_name} - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 准确率曲线
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_title(f'{model_name} {base_model_name} - Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 学习率曲线
    ax3.plot(epochs, history['learning_rates'], 'g-', label='Learning Rate', linewidth=2)
    ax3.set_title(f'{model_name} {base_model_name} - Learning Rate')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 每个epoch的训练时间
    ax4.plot(epochs, history['epoch_times'], 'purple', label='Epoch Time', linewidth=2)
    ax4.set_title(f'{model_name} {base_model_name} - Epoch Training Time')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Time (seconds)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图像
    plot_path = os.path.join(plot_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"训练曲线已保存: {plot_path}")


# 评估SSCNN模型
def evaluate_sscnn(model, dataloader, criterion, return_preds=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for left_imgs, right_imgs, _, _, labels in dataloader:
            left_imgs = left_imgs.to(model.device)
            right_imgs = right_imgs.to(model.device)
            labels = labels.to(model.device)

            outputs = model(left_imgs, right_imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if return_preds:
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total

    if return_preds:
        return avg_loss, accuracy, all_labels, all_preds
    else:
        return avg_loss, accuracy


# 评估RSSCNN模型
def evaluate_rsscnn(model, dataloader, return_preds=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for left_imgs, right_imgs, _, _, labels in dataloader:
            left_imgs = left_imgs.to(model.device)
            right_imgs = right_imgs.to(model.device)
            labels = labels.to(model.device)

            class_out, rank1, rank2 = model(left_imgs, right_imgs)
            loss = model.compute_loss(class_out, rank1, rank2, labels)

            total_loss += loss.item()
            _, predicted = torch.max(class_out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if return_preds:
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total

    if return_preds:
        return avg_loss, accuracy, all_labels, all_preds
    return avg_loss, accuracy


def run_sscnn_training(cfg, dataset):
    """运行SSCNN训练"""
    print("Training SSCNN with ...", cfg.base_model_name)
    early_stopper_sscnn = SchedulerEarlyStopper(max_plateaus=cfg.max_lr_plateaus)

    sscnn_model, sscnn_history, output_dir = train_sscnn(
        model_name=cfg.base_model_name,
        num_epochs=cfg.num_epochs,
        lr=cfg.learning_rate,
        batch_size=cfg.batch_size,
        early_stopper=early_stopper_sscnn,
        device=cfg.device,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        factor=cfg.factor,
        patience=cfg.patience,
        my_dataset=dataset
    )

    print(f"SSCNN训练完成！结果保存在: {output_dir}")
    return sscnn_model, sscnn_history, output_dir


def run_rsscnn_training(cfg, dataset_1):
    """运行RSSCNN训练"""
    print("\nTraining RSSCNN with", cfg.base_model_name)
    early_stopper_rsscnn = SchedulerEarlyStopper(max_plateaus=cfg.max_lr_plateaus)

    rsscnn_model, rsscnn_history, output_dir = train_rsscnn(
        model_name=cfg.base_model_name,
        num_epochs=cfg.num_epochs,
        lr=cfg.learning_rate,
        lambda_r=cfg.lambda_r,
        batch_size=cfg.batch_size,
        early_stopper=early_stopper_rsscnn,
        device=cfg.device,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        factor=cfg.factor,
        patience=cfg.patience,
        my_dataset_1=dataset_1
    )

    print(f"RSSCNN训练完成！结果保存在: {output_dir}")
    return rsscnn_model, rsscnn_history, output_dir


if __name__ == "__main__":
    # 加载配置
    cfg = config.config_image_model.Config()

    # 创建数据集
    dataset_1 = MyPP2Dataset(transform=cfg.transform, is_flipped=False)

    # 选择要训练的模型
    train_sscnn_flag = True  # 设置为True训练SSCNN，False则不训练
    train_rsscnn_flag = False  # 设置为True训练RSSCNN，False则不训练

    # 训练SSCNN模型
    if train_sscnn_flag:
        sscnn_model, sscnn_history, sscnn_dir = run_sscnn_training(cfg, dataset_1)

    # 训练RSSCNN模型
    if train_rsscnn_flag:
        rsscnn_model, rsscnn_history, rsscnn_dir = run_rsscnn_training(cfg, dataset_1)

    print("所有训练任务完成！")