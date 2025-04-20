import os
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

import config.config_image_model
from MyPP2Dataset import MyPP2Dataset, create_dataloaders
from models.image_models import SSCNN, RSSCNN
from utils.plot_training_curve import live_plot
from utils.save_confusion_matrix import save_confusion_matrix
import numpy as np
from tqdm import tqdm

from utils.my_transforms import transform_cnn_2
from utils.early_stop import SchedulerEarlyStopper

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)


# 训练SSCNN模型
def train_sscnn(model_name='AlexNet', num_epochs=300, lr=0.001, batch_size=4, early_stopper=None, device=None,
                my_dataset=None):
    train_loader, val_loader = create_dataloaders(my_dataset, train_ratio=0.8, batch_size=batch_size, shuffle=True)
    model = SSCNN(base_model_name=model_name).to(device)
    model.device = device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for left_imgs, right_imgs, _, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
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

        scheduler.step(val_loss)

        # 确保文件夹存在
        os.makedirs(f"outputs/models/sscnn", exist_ok=True)

        # 保存混淆矩阵
        save_confusion_matrix(val_labels, val_preds,
                              class_names=['Left', 'Right'],
                              title=f"SSCNN {model_name} - Epoch {epoch + 1}",
                              filename=f"outputs/figures/sscnn/{model_name}/confmat_epoch{epoch + 1}.png",
                              matrix_path=f"outputs/figures/sscnn/{model_name}/confmat_epoch{epoch + 1}"
                              )

        # 实时绘图
        live_plot(history)

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'outputs/models/sscnn/best_sscnn_{}.pth'.format(model_name))
            # 保存测试集准确率最高时的混淆矩阵
            save_confusion_matrix(val_labels, val_preds,
                                  class_names=['Left', 'Right'],
                                  title=f"SSCNN {model_name} - best epoch",
                                  filename=f"outputs/figures/sscnn/{model_name}/confmat_best.png",
                                  matrix_path=f"outputs/figures/sscnn/{model_name}/confmat_best"
                                  )

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # 判断是否需要提前停止，一定要在最后早停，不然会少保存一次
        if early_stopper.step(optimizer):
            print("训练提前终止：学习率已经衰减达到最大次数。")
            break
    plt.close()  # 关闭动态图窗口
    return model, history


# 训练RSSCNN模型
def train_rsscnn(model_name='AlexNet', num_epochs=300, lr=0.001, lambda_r=0.1, batch_size=4,
                 early_stopper=None, device=None, my_dataset=None):
    train_loader, val_loader = create_dataloaders(my_dataset, train_ratio=0.8, batch_size=batch_size, shuffle=True)
    model = RSSCNN(base_model_name=model_name, lambda_r=lambda_r).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for left_imgs, right_imgs, _, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
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

        scheduler.step(val_loss)

        # 确保文件夹存在
        os.makedirs('outputs/models/rsscnn', exist_ok=True)

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'outputs/models/rsscnn/best_rsscnn_{}.pth'.format(model_name))
            # 保存测试集准确率最高时的混淆矩阵
            save_confusion_matrix(val_labels, val_preds, class_names=['Left', 'Right'],
                                  title=f"RSSCNN {model_name} - best epoch",
                                  filename=f"outputs/figures/rsscnn/{model_name}/confmat_best.png",
                                  matrix_path=f"outputs/figures/rsscnn/{model_name}/confmat_best"
                                  )

        # 混淆矩阵保存
        save_confusion_matrix(val_labels, val_preds, class_names=['Left', 'Right'],
                              title=f"RSSCNN {model_name} - Epoch {epoch + 1}",
                              filename=f"outputs/figures/rsscnn/{model_name}/confmat_epoch{epoch + 1}.png",
                              matrix_path=f"outputs/figures/rsscnn/{model_name}/confmat_epoch{epoch + 1}"
                              )

        # 实时训练曲线绘图
        live_plot(history)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # 判断是否需要提前停止，一定要在最后早停，不然会少保存一次
        if early_stopper.step(optimizer):
            print("训练提前终止：学习率已经衰减达到最大次数。")
            break
    plt.close()  # 关闭动态图窗口
    return model, history


# 评估SSCNN模型
def evaluate_sscnn(model, dataloader, criterion, return_preds=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for left_imgs, right_imgs, _, labels in dataloader:
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
        for left_imgs, right_imgs, _, labels in dataloader:
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


if __name__ == "__main__":
    # 创建数据集
    dataset = MyPP2Dataset(is_flipped=False, transform=transform_cnn_2)

    cfg = config.config_image_model.Config()

    # 训练SSCNN模型  ---已经跑通，没问题
    print("Training SSCNN with ...", cfg.base_model_name)
    early_stopper_sscnn = SchedulerEarlyStopper(max_plateaus=cfg.max_lr_plateaus)  # 初始化早停器

    sscnn_model, sscnn_history = train_sscnn(
        model_name=cfg.base_model_name,
        num_epochs=cfg.num_epochs,
        lr=cfg.learning_rate,
        batch_size=cfg.batch_size,
        early_stopper=early_stopper_sscnn,
        device=cfg.device,
        my_dataset=dataset
    )

    live_plot(sscnn_history,
              title="SSCNN Training Curve",
              save_path=f"outputs/training_curve/sscnn/{cfg.base_model_name}/training_curve.png", )

    # 训练RSSCNN模型
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
        my_dataset=dataset
    )

    live_plot(sscnn_history,
              title="RSSCNN Training Curve",
              save_path=f"outputs/training_curve/rsscnn/{cfg.base_model_name}/training_curve.png", )
