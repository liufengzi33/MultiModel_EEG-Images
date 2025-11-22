import torch
import torch.optim as optim
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
from MyPP2Dataset import MyPP2Dataset, create_subject_dataloaders
from models.multi_model import MultiModalFusionNetwork
from config import config_multi_model
import sys
import os
from utils.early_stop import SchedulerEarlyStopper


def train_multimodal_with_config():
    """使用配置类训练多模态融合网络"""

    # 初始化配置
    config = config_multi_model.Config()
    print(f"使用设备: {config.device}")

    # 创建数据集
    print("创建数据集...")
    dataset_normal = MyPP2Dataset(
        is_flipped=False,
        transform=config.transform,
        subject_id=config.subject_id  # 根据实际情况修改被试ID
    )

    dataset_flipped = MyPP2Dataset(
        is_flipped=True,
        transform=config.transform,
        subject_id=config.subject_id  # 根据实际情况修改被试ID
    )

    # 创建数据加载器
    train_loader, test_loader = create_subject_dataloaders(
        dataset_1=dataset_normal,
        dataset_2=dataset_flipped,
        batch_size=config.batch_size,
        shuffle=True
    )

    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")
    print(f"训练批次数: {len(train_loader)}")
    print(f"测试批次数: {len(test_loader)}")

    # 初始化模型
    print("初始化模型...")
    model = MultiModalFusionNetwork(
        eeg_model_name=config.base_eeg_model,
        image_model_name=config.base_image_model,
        image_model_type=config.image_model_type,
        in_chans=64,
        n_classes=2,
        input_window_samples=2000,
        use_pretrained_eeg=config.use_pretrained,
        use_pretrained_image=config.use_pretrained,
        common_dim=512,
        private_dim=256,
        dropout_rate=0.5,
        alpha=config.alpha,
        beta=config.beta
    ).to(config.device)

    # 打印模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.factor,
        patience=config.patience,
        verbose=True
    )

    # 初始化早停器
    early_stopper = SchedulerEarlyStopper(max_plateaus=config.max_lr_plateaus)

    # 训练状态跟踪
    best_val_loss = float('inf')
    lr_plateau_count = 0
    best_model_state = None

    # 训练历史记录
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'task_loss': [], 'common_sim_loss': [], 'private_diff_loss': [],
        'learning_rates': []
    }

    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"checkpoints/multimodal_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    print("开始训练...")
    start_time = time.time()

    for epoch in range(config.num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        epoch_task_loss = 0.0
        epoch_common_loss = 0.0
        epoch_private_loss = 0.0

        for batch_idx, (left_img, right_img, left_eeg, right_eeg, labels) in enumerate(train_loader):
            # 移动到设备
            left_img = left_img.to(config.device)
            right_img = right_img.to(config.device)
            left_eeg = left_eeg.to(config.device)
            right_eeg = right_eeg.to(config.device)
            labels = labels.to(config.device)

            # 前向传播和损失计算
            optimizer.zero_grad()
            logits, eeg_common, image_common, eeg_private, image_private = model(
                left_eeg, right_eeg, left_img, right_img
            )

            losses = model.compute_loss(
                eeg_common, image_common, eeg_private, image_private, logits, labels
            )

            total_loss = losses['total_loss']

            # 反向传播
            total_loss.backward()
            optimizer.step()

            # 统计信息
            train_loss += total_loss.item()
            epoch_task_loss += losses['task_loss'].item()
            epoch_common_loss += losses['common_sim_loss'].item()
            epoch_private_loss += losses['private_diff_loss'].item()

            # 计算准确率
            if logits.dim() == 1:
                preds = (torch.sigmoid(logits) > 0.5).long()
            else:
                preds = torch.argmax(logits, dim=1)

            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for left_img, right_img, left_eeg, right_eeg, labels in test_loader:
                left_img = left_img.to(config.device)
                right_img = right_img.to(config.device)
                left_eeg = left_eeg.to(config.device)
                right_eeg = right_eeg.to(config.device)
                labels = labels.to(config.device)

                logits, eeg_common, image_common, eeg_private, image_private = model(
                    left_eeg, right_eeg, left_img, right_img
                )

                losses = model.compute_loss(
                    eeg_common, image_common, eeg_private, image_private, logits, labels
                )

                val_loss += losses['total_loss'].item()

                if logits.dim() == 1:
                    preds = (torch.sigmoid(logits) > 0.5).long()
                else:
                    preds = torch.argmax(logits, dim=1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        # 计算平均指标
        train_loss /= len(train_loader)
        val_loss /= len(test_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total

        epoch_task_loss /= len(train_loader)
        epoch_common_loss /= len(train_loader)
        epoch_private_loss /= len(train_loader)

        # 学习率调度
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        # 使用早停器检查是否应该停止
        should_stop = early_stopper.step(optimizer)

        # 记录学习率变化
        if new_lr < current_lr:
            lr_plateau_count += 1
            print(f"学习率衰减 #{lr_plateau_count}: {current_lr:.2e} -> {new_lr:.2e}")

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['task_loss'].append(epoch_task_loss)
        history['common_sim_loss'].append(epoch_common_loss)
        history['private_diff_loss'].append(epoch_private_loss)
        history['learning_rates'].append(new_lr)

        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'\nEpoch {epoch + 1}/{config.num_epochs}:')
            print(f'  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
            print(f'  验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%')
            print(
                f'  损失分量 - 任务: {epoch_task_loss:.4f}, 公共: {epoch_common_loss:.4f}, 私有: {epoch_private_loss:.4f}')
            print(f'  学习率: {new_lr:.2e}, 学习率衰减次数: {early_stopper.plateau_count}/{config.max_lr_plateaus}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_acc': val_acc,
                'config': config.__dict__,
                'history': history
            }, os.path.join(save_dir, 'best_model.pth'))

            if (epoch + 1) % 10 == 0:
                print(f'  ✅ 保存最佳模型，验证损失: {val_loss:.4f}')

        # 检查早停条件
        if should_stop:
            print(f"\n达到最大学习率衰减次数 {config.max_lr_plateaus}，停止训练")
            break

    # 训练结束
    training_time = time.time() - start_time
    print(f"\n训练完成! 总时间: {training_time / 60:.2f} 分钟")
    print(f"总训练轮次: {epoch + 1}")
    print(f"最终学习率衰减次数: {early_stopper.plateau_count}")

    # 保存最终模型
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config.__dict__,
        'history': history,
        'final_plateau_count': early_stopper.plateau_count
    }, os.path.join(save_dir, 'final_model.pth'))

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("已加载最佳模型")

    # 绘制训练曲线
    plot_training_history(history, save_dir)

    print(f"所有文件保存在: {save_dir}")
    return model, history, config


def plot_training_history(history, save_dir):
    """绘制训练历史曲线"""
    plt.figure(figsize=(15, 12))

    epochs = range(1, len(history['train_loss']) + 1)

    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], label='训练损失', linewidth=2)
    plt.plot(epochs, history['val_loss'], label='验证损失', linewidth=2)
    plt.title('总损失曲线', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['train_acc'], label='训练准确率', linewidth=2)
    plt.plot(epochs, history['val_acc'], label='验证准确率', linewidth=2)
    plt.title('准确率曲线', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 损失分量
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['task_loss'], label='任务损失', linewidth=2)
    plt.plot(epochs, history['common_sim_loss'], label='公共相似损失', linewidth=2)
    plt.plot(epochs, history['private_diff_loss'], label='私有差异损失', linewidth=2)
    plt.title('损失分量', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 学习率变化
    plt.subplot(2, 2, 4)
    plt.semilogy(epochs, history['learning_rates'], color='purple', linewidth=2)
    plt.title('学习率变化', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate (log)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_model(model, config):
    """评估模型性能"""
    # 创建测试数据集
    dataset_normal = MyPP2Dataset(
        is_flipped=False,
        transform=config.transform,
        subject_id="01gh"
    )

    dataset_flipped = MyPP2Dataset(
        is_flipped=True,
        transform=config.transform,
        subject_id="01gh"
    )

    _, test_loader = create_subject_dataloaders(
        dataset_1=dataset_normal,
        dataset_2=dataset_flipped,
        batch_size=config.batch_size,
        shuffle=False
    )

    # 测试模型
    model.eval()
    all_preds = []
    all_labels = []
    test_loss = 0.0

    with torch.no_grad():
        for left_img, right_img, left_eeg, right_eeg, labels in test_loader:
            left_img = left_img.to(config.device)
            right_img = right_img.to(config.device)
            left_eeg = left_eeg.to(config.device)
            right_eeg = right_eeg.to(config.device)
            labels = labels.to(config.device)

            logits, eeg_common, image_common, eeg_private, image_private = model(
                left_eeg, right_eeg, left_img, right_img
            )

            losses = model.compute_loss(
                eeg_common, image_common, eeg_private, image_private, logits, labels
            )

            test_loss += losses['total_loss'].item()

            if logits.dim() == 1:
                preds = (torch.sigmoid(logits) > 0.5).long()
            else:
                preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss /= len(test_loader)
    accuracy = 100. * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)

    print(f"\n最终测试结果:")
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {accuracy:.2f}%")

    return test_loss, accuracy, all_preds, all_labels


if __name__ == "__main__":
    # 开始训练
    trained_model, training_history, training_config = train_multimodal_with_config()

    # 评估模型
    test_loss, test_accuracy, predictions, true_labels = evaluate_model(trained_model, training_config)