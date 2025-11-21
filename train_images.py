import os
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

import config.config_image_model
from MyPP2Dataset import MyPP2Dataset, create_dataloaders, create_subject_dataloaders
from models.image_models import SSCNN, RSSCNN
from utils.save_confusion_matrix import save_confusion_matrix
from tqdm import tqdm

from utils.early_stop import SchedulerEarlyStopper


# 设置随机种子保证可重复性
# torch.manual_seed(42)
# np.random.seed(42)


# 训练SSCNN模型
def train_sscnn(model_name='AlexNet', num_epochs=300, lr=0.001, batch_size=4, early_stopper=None,
                momentum=0.9,
                factor=0.1,
                patience=10,
                device=None,
                my_dataset=None):
    train_loader, val_loader = create_dataloaders(my_dataset, batch_size=batch_size, shuffle=True)
    model = SSCNN(base_model_name=model_name).to(device)
    model.device = device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=factor, patience=patience)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    current_lr = lr

    # 创建图形和坐标轴
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    plt.ion()  # 开启交互模式

    for epoch in range(num_epochs):
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
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        val_loss, val_acc, val_labels, val_preds = evaluate_sscnn(model, val_loader, criterion, return_preds=True)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 在调度器step之前保存旧的学习率
        old_lr = current_lr
        scheduler.step(val_loss)
        # 获取新的学习率
        current_lr = optimizer.param_groups[0]['lr']
        # 检查学习率是否变化
        if current_lr != old_lr:
            print(f"学习率更新: {old_lr:.6f} -> {current_lr:.6f}")

        # 确保文件夹存在
        os.makedirs(f"outputs/outputs_images/models/sscnn", exist_ok=True)

        # 保存混淆矩阵
        save_confusion_matrix(val_labels, val_preds,
                              class_names=['Left', 'Right'],
                              title=f"SSCNN {model_name} - Epoch {epoch + 1}",
                              filename=f"outputs/outputs_images/figures/sscnn/{model_name}/confmat_epoch{epoch + 1}.png",
                              matrix_path=f"outputs/outputs_images/figures/sscnn/{model_name}/confmat_epoch{epoch + 1}"
                              )

        # 实时绘图 - 更新同一个图
        update_live_plot(history, fig, ax1, ax2, model_name="SSCNN")

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'outputs/outputs_images/models/sscnn/best_sscnn_{}.pth'.format(model_name))
            # 保存测试集准确率最高时的混淆矩阵
            save_confusion_matrix(val_labels, val_preds,
                                  class_names=['Left', 'Right'],
                                  title=f"SSCNN {model_name} - best epoch",
                                  filename=f"outputs/outputs_images/figures/sscnn/{model_name}/confmat_best.png",
                                  matrix_path=f"outputs/outputs_images/figures/sscnn/{model_name}/confmat_best"
                                  )

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # 判断是否需要提前停止，一定要在最后早停，不然会少保存一次
        if early_stopper.step(optimizer):
            print("训练提前终止：学习率已经衰减达到最大次数。")
            break

    plt.ioff()  # 关闭交互模式
    plt.close()  # 关闭动态图窗口

    # 保存最终训练曲线
    save_final_plot(history, model_name="SSCNN",
                    save_path=f"outputs/outputs_images/training_curve/sscnn/{model_name}/training_curve.png")

    return model, history


# 训练RSSCNN模型
def train_rsscnn(model_name='AlexNet', num_epochs=300, lr=0.001, lambda_r=0.1, batch_size=4,
                 early_stopper=None,
                 momentum=0.9,
                 factor=0.1,
                 patience=10,
                 device=None, my_dataset_1=None, my_dataset_2=None, ):
    # train_loader, val_loader = create_subject_dataloaders(my_dataset_1,my_dataset_2, batch_size=batch_size, shuffle=True)
    train_loader, val_loader = create_dataloaders(my_dataset_1, batch_size=batch_size, shuffle=True)
    model = RSSCNN(base_model_name=model_name, lambda_r=lambda_r).to(device)
    model.device = device
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=factor, patience=patience)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    current_lr = lr  # 跟踪当前学习率

    # 创建图形和坐标轴
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    plt.ion()  # 开启交互模式

    for epoch in range(num_epochs):
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
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # 验证阶段 + 混淆矩阵 + 动态绘图
        val_loss, val_acc, val_labels, val_preds = evaluate_rsscnn(model, val_loader, return_preds=True)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 在调度器step之前保存旧的学习率
        old_lr = current_lr
        scheduler.step(val_loss)
        # 获取新的学习率
        current_lr = optimizer.param_groups[0]['lr']
        # 检查学习率是否变化
        if current_lr != old_lr:
            print(f"学习率更新: {old_lr:.6f} -> {current_lr:.6f}")

        # 确保文件夹存在
        os.makedirs('outputs/outputs_images/models/rsscnn', exist_ok=True)

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'outputs/outputs_images/models/rsscnn/best_rsscnn_{}.pth'.format(model_name))
            # 保存测试集准确率最高时的混淆矩阵
            save_confusion_matrix(val_labels, val_preds, class_names=['Left', 'Right'],
                                  title=f"RSSCNN {model_name} - best epoch",
                                  filename=f"outputs/outputs_images/figures/rsscnn/{model_name}/confmat_best.png",
                                  matrix_path=f"outputs/outputs_images/figures/rsscnn/{model_name}/confmat_best"
                                  )

        # 混淆矩阵保存
        save_confusion_matrix(val_labels, val_preds, class_names=['Left', 'Right'],
                              title=f"RSSCNN {model_name} - Epoch {epoch + 1}",
                              filename=f"outputs/outputs_images/figures/rsscnn/{model_name}/confmat_epoch{epoch + 1}.png",
                              matrix_path=f"outputs/outputs_images/figures/rsscnn/{model_name}/confmat_epoch{epoch + 1}"
                              )

        # 实时训练曲线绘图 - 更新同一个图
        update_live_plot(history, fig, ax1, ax2, model_name="RSSCNN")

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # 判断是否需要提前停止，一定要在最后早停，不然会少保存一次
        if early_stopper.step(optimizer):
            print("训练提前终止：学习率已经衰减达到最大次数。")
            break

    plt.ioff()  # 关闭交互模式
    plt.close()  # 关闭动态图窗口

    # 保存最终训练曲线
    save_final_plot(history, model_name="RSSCNN",
                    save_path=f"outputs/outputs_images/training_curve/rsscnn/{model_name}/training_curve.png")

    return model, history


def update_live_plot(history, fig, ax1, ax2, model_name="Model"):
    """更新实时训练曲线图"""
    # 清空当前图形
    ax1.clear()
    ax2.clear()

    epochs = range(1, len(history['train_loss']) + 1)

    # 绘制损失曲线
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_title(f'{model_name} - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # 绘制准确率曲线
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    ax2.set_title(f'{model_name} - Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    # 调整布局
    plt.tight_layout()

    # 更新图形
    plt.draw()
    plt.pause(0.1)  # 短暂暂停以更新图形


def save_final_plot(history, model_name="Model", save_path=None):
    """保存最终训练曲线图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # 绘制损失曲线
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_title(f'{model_name} - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # 绘制准确率曲线
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    ax2.set_title(f'{model_name} - Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    # 确保保存目录存在
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


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
    early_stopper_sscnn = SchedulerEarlyStopper(max_plateaus=cfg.max_lr_plateaus)  # 初始化早停器

    sscnn_model, sscnn_history = train_sscnn(
        model_name=cfg.base_model_name,
        num_epochs=cfg.num_epochs,
        lr=cfg.learning_rate,
        batch_size=cfg.batch_size,
        early_stopper=early_stopper_sscnn,
        device=cfg.device,
        momentum=cfg.momentum,
        factor=cfg.factor,
        patience=cfg.patience,
        my_dataset=dataset
    )

    print("SSCNN训练完成！")
    return sscnn_model, sscnn_history


def run_rsscnn_training(cfg, dataset_1, dataset_2):
    """运行RSSCNN训练"""
    print("\nTraining RSSCNN with", cfg.base_model_name)
    early_stopper_rsscnn = SchedulerEarlyStopper(max_plateaus=cfg.max_lr_plateaus)  # 初始化早停器

    rsscnn_model, rsscnn_history = train_rsscnn(
        model_name=cfg.base_model_name,
        num_epochs=cfg.num_epochs,
        lr=cfg.learning_rate,
        lambda_r=cfg.lambda_r,
        batch_size=cfg.batch_size,
        early_stopper=early_stopper_rsscnn,
        device=cfg.device,
        momentum=cfg.momentum,
        factor=cfg.factor,
        patience=cfg.patience,
        my_dataset_1=dataset_1,
        my_dataset_2=dataset_2
    )

    print("RSSCNN训练完成！")
    return rsscnn_model, rsscnn_history


if __name__ == "__main__":
    # 加载配置
    cfg = config.config_image_model.Config()

    # 创建数据集
    dataset_1 = MyPP2Dataset(transform=cfg.transform, is_flipped=False)
    dataset_2 = MyPP2Dataset(transform=cfg.transform, is_flipped=True)

    # 选择要训练的模型
    train_sscnn_flag = False  # 设置为True训练SSCNN，False则不训练
    train_rsscnn_flag = True  # 设置为True训练RSSCNN，False则不训练

    # 训练SSCNN模型
    if train_sscnn_flag:
        sscnn_model, sscnn_history = run_sscnn_training(cfg, dataset_1)
        # 可以在这里添加SSCNN的后续处理，如测试、保存结果等

    # 训练RSSCNN模型
    if train_rsscnn_flag:
        rsscnn_model, rsscnn_history = run_rsscnn_training(cfg, dataset_1, dataset_2)
        # 可以在这里添加RSSCNN的后续处理，如测试、保存结果等

    print("所有训练任务完成！")