import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
import json
import pandas as pd
from datetime import datetime
from sklearn.metrics import f1_score, roc_auc_score

import config.config_image_model
from MyPP2Dataset import MyPP2Dataset, create_dataloaders
from models.image_models import SSCNN, RSSCNN
from tqdm import tqdm
import seaborn as sns
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

        # 子目录 (已移除混淆矩阵目录)
        self.model_dir = os.path.join(self.base_dir, "models")
        self.plot_dir = os.path.join(self.base_dir, "plots")
        self.data_dir = os.path.join(self.base_dir, "training_data")
        self.checkpoints_dir = os.path.join(self.base_dir, "checkpoints")

        # 创建所有目录
        for directory in [self.model_dir, self.plot_dir, self.data_dir, self.checkpoints_dir]:
            os.makedirs(directory, exist_ok=True)

        print(f"输出目录结构:")
        print(f"  - 基础目录: {self.base_dir}")
        print(f"  - 模型目录: {self.model_dir}")
        print(f"  - 检查点目录: {self.checkpoints_dir}")
        print(f"  - 图像目录: {self.plot_dir}")
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

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=factor, patience=patience)

    # 更新历史记录以包含 F1 和 AUC
    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [], 'train_auc': [], 'val_auc': [],
        'learning_rates': [], 'epoch_times': [], 'best_epoch': 0,
        'checkpoint_epochs': []
    }

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

        train_labels_list = []
        train_preds_list = []
        train_probs_list = []

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
            probs = F.softmax(outputs.data, dim=1)[:, 1]  # 提取正类别的概率
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            train_labels_list.extend(labels.cpu().numpy())
            train_preds_list.extend(predicted.cpu().numpy())
            train_probs_list.extend(probs.cpu().numpy())

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # 计算训练集的 F1 和 AUC
        train_f1 = f1_score(train_labels_list, train_preds_list, average='macro')
        try:
            train_auc = roc_auc_score(train_labels_list, train_probs_list)
        except ValueError:
            train_auc = 0.5  # 防止 batch 内只有单一类别导致报错

        val_loss, val_acc, val_f1, val_auc = evaluate_sscnn(model, val_loader, criterion)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        history['learning_rates'].append(current_lr)
        history['epoch_times'].append((datetime.now() - epoch_start_time).total_seconds())

        # 学习率更新
        old_lr = current_lr
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != old_lr:
            print(f"学习率更新: {old_lr:.6f} -> {current_lr:.6f}")

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
            print(f"✅ 保存最佳验证模型 - 准确率: {val_acc:.2f}%, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")

        # 2. 每10个epoch保存检查点
        if (epoch + 1) % 10 == 0 or epoch == 0:
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

        print(f'Epoch [{epoch + 1}/{num_epochs}] | '
              f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_f1:.4f}, AUC: {train_auc:.4f} | '
              f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.4f}, AUC: {val_auc:.4f}')

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

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=factor, patience=patience)

    # 更新历史记录以包含 F1 和 AUC
    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [], 'train_auc': [], 'val_auc': [],
        'learning_rates': [], 'epoch_times': [], 'best_epoch': 0,
        'checkpoint_epochs': []
    }

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

        train_labels_list = []
        train_preds_list = []
        train_probs_list = []

        for left_imgs, right_imgs, _, _, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            left_imgs, right_imgs, labels = left_imgs.to(device), right_imgs.to(device), labels.to(device)

            class_out, rank1, rank2 = model(left_imgs, right_imgs)
            loss = model.compute_loss(class_out, rank1, rank2, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            probs = F.softmax(class_out.data, dim=1)[:, 1]
            _, predicted = torch.max(class_out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            train_labels_list.extend(labels.cpu().numpy())
            train_preds_list.extend(predicted.cpu().numpy())
            train_probs_list.extend(probs.cpu().numpy())

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        train_f1 = f1_score(train_labels_list, train_preds_list, average='macro')
        try:
            train_auc = roc_auc_score(train_labels_list, train_probs_list)
        except ValueError:
            train_auc = 0.5

        val_loss, val_acc, val_f1, val_auc = evaluate_rsscnn(model, val_loader)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        history['learning_rates'].append(current_lr)
        history['epoch_times'].append((datetime.now() - epoch_start_time).total_seconds())

        # 学习率更新
        old_lr = current_lr
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != old_lr:
            print(f"学习率更新: {old_lr:.6f} -> {current_lr:.6f}")

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
            print(f"✅ 保存最佳验证模型 - 准确率: {val_acc:.2f}%, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")

        # 2. 每10个epoch保存检查点
        if (epoch + 1) % 10 == 0 or epoch == 0:
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

        print(f"Epoch [{epoch + 1}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_f1:.4f}, AUC: {train_auc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")

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
        'train_f1': history['train_f1'],
        'val_f1': history['val_f1'],
        'train_auc': history['train_auc'],
        'val_auc': history['val_auc'],
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
        'best_val_f1': max(history['val_f1']) if history['val_f1'] else 0,
        'best_val_auc': max(history['val_auc']) if history['val_auc'] else 0,
        'total_training_time_seconds': sum(history['epoch_times']),
        'total_epochs': len(history['train_loss']),
        'checkpoint_epochs': history['checkpoint_epochs']
    }

    summary_path = os.path.join(trainer.data_dir, "training_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 4. 绘制并保存所有训练曲线
    plot_all_curves(history, model_name, train_config['base_model_name'], trainer.plot_dir)

    print(f"\n训练结果已保存到: {trainer.base_dir}")
    print(f"最佳验证准确率: {summary['best_val_accuracy']:.2f}% (第 {summary['best_epoch']} 轮)")
    print(f"总训练时间: {summary['total_training_time_seconds']:.2f} 秒")


def plot_all_curves(history, model_name, base_model_name, plot_dir):
    """
    分别绘制并保存三个图表（Seaborn 学术风格）:
    1. Loss和Accuracy (双子图)
    2. F1 Score (单图)
    3. AUC-ROC (单图)
    """

    # 设置 seaborn 学术风格
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    epochs = range(1, len(history['train_loss']) + 1)

    # ==========================
    # 图 1: Loss 和 Accuracy (双子图)
    # ==========================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss 子图
    ax1.plot(epochs, history['train_loss'], linestyle='-', marker='o', markersize=4, label='Train Loss', linewidth=2,
             alpha=0.8)
    ax1.plot(epochs, history['val_loss'], linestyle='-', marker='s', markersize=4, label='Val Loss', linewidth=2,
             alpha=0.8)
    ax1.set_title(f'{model_name} {base_model_name} - Loss', fontweight='bold')
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.legend(loc='upper right', frameon=True, shadow=True)

    # Accuracy 子图
    ax2.plot(epochs, history['train_acc'], linestyle='-', marker='o', markersize=4, label='Train Acc', linewidth=2,
             alpha=0.8)
    ax2.plot(epochs, history['val_acc'], linestyle='-', marker='s', markersize=4, label='Val Acc', linewidth=2,
             alpha=0.8)
    ax2.set_title(f'{model_name} {base_model_name} - Accuracy', fontweight='bold')
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontweight='bold')
    ax2.legend(loc='lower right', frameon=True, shadow=True)

    plt.tight_layout()
    loss_acc_path = os.path.join(plot_dir, "loss_acc_curves.png")
    plt.savefig(loss_acc_path, dpi=300, bbox_inches='tight')
    plt.close()

    # ==========================
    # 图 2: F1 分数 (单独的图)
    # ==========================
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history['train_f1'], linestyle='-', marker='o', markersize=4, label='Train F1', linewidth=2,
             alpha=0.8)
    plt.plot(epochs, history['val_f1'], linestyle='-', marker='s', markersize=4, label='Val F1', linewidth=2, alpha=0.8)
    plt.title(f'{model_name} {base_model_name} - F1 Score', fontweight='bold')
    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('F1 Score', fontweight='bold')
    plt.legend(loc='lower right', frameon=True, shadow=True)

    plt.tight_layout()
    f1_path = os.path.join(plot_dir, "f1_curve.png")
    plt.savefig(f1_path, dpi=300, bbox_inches='tight')
    plt.close()

    # ==========================
    # 图 3: AUC-ROC (单独的图)
    # ==========================
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history['train_auc'], linestyle='-', marker='o', markersize=4, label='Train AUC', linewidth=2,
             alpha=0.8)
    plt.plot(epochs, history['val_auc'], linestyle='-', marker='s', markersize=4, label='Val AUC', linewidth=2,
             alpha=0.8)
    plt.title(f'{model_name} {base_model_name} - AUC-ROC', fontweight='bold')
    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('AUC-ROC', fontweight='bold')
    plt.legend(loc='lower right', frameon=True, shadow=True)

    plt.tight_layout()
    auc_path = os.path.join(plot_dir, "auc_roc_curve.png")
    plt.savefig(auc_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 重置 matplotlib 风格，避免影响后续代码的其他绘图
    sns.reset_orig()

    print(f"✅ 三个绘图已保存:\n  1. {loss_acc_path}\n  2. {f1_path}\n  3. {auc_path}")

# 评估SSCNN模型
def evaluate_sscnn(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for left_imgs, right_imgs, _, _, labels in dataloader:
            left_imgs = left_imgs.to(model.device)
            right_imgs = right_imgs.to(model.device)
            labels = labels.to(model.device)

            outputs = model(left_imgs, right_imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            probs = F.softmax(outputs.data, dim=1)[:, 1]
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total

    f1 = f1_score(all_labels, all_preds, average='macro')
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5

    return avg_loss, accuracy, f1, auc


# 评估RSSCNN模型
def evaluate_rsscnn(model, dataloader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for left_imgs, right_imgs, _, _, labels in dataloader:
            left_imgs = left_imgs.to(model.device)
            right_imgs = right_imgs.to(model.device)
            labels = labels.to(model.device)

            class_out, rank1, rank2 = model(left_imgs, right_imgs)
            loss = model.compute_loss(class_out, rank1, rank2, labels)

            total_loss += loss.item()
            probs = F.softmax(class_out.data, dim=1)[:, 1]
            _, predicted = torch.max(class_out.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total

    f1 = f1_score(all_labels, all_preds, average='macro')
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5

    return avg_loss, accuracy, f1, auc


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


def run_rsscnn_training(cfg, dataset):
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
        my_dataset_1=dataset
    )

    print(f"RSSCNN训练完成！结果保存在: {output_dir}")
    return rsscnn_model, rsscnn_history, output_dir


if __name__ == "__main__":
    # 加载配置
    cfg = config.config_image_model.Config()

    # 创建数据集
    dataset = MyPP2Dataset(transform=cfg.transform, is_flipped=False)

    # 选择要训练的模型
    train_sscnn_flag = True  # 设置为True训练SSCNN，False则不训练
    train_rsscnn_flag = True  # 设置为True训练RSSCNN，False则不训练

    # 训练SSCNN模型
    if train_sscnn_flag:
        sscnn_model, sscnn_history, sscnn_dir = run_sscnn_training(cfg, dataset)

    # 训练RSSCNN模型
    if train_rsscnn_flag:
        rsscnn_model, rsscnn_history, rsscnn_dir = run_rsscnn_training(cfg, dataset)

    print("所有训练任务完成！")