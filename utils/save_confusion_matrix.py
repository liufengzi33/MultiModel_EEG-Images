import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import pandas as pd

def save_confusion_matrix(y_true, y_pred, class_names, title, filename, matrix_path=None, normalize=False):
    """
    绘制并保存混淆矩阵图像，并将混淆矩阵数值保存为 .npy 和 .csv 格式。

    Args:
        y_true (list or ndarray): 真实标签
        y_pred (list or ndarray): 预测标签
        class_names (list): 类别名称
        title (str): 图标题
        filename (str): 图像保存路径
        matrix_path (str): 混淆矩阵保存路径（不含扩展名）
        normalize (bool): 是否归一化混淆矩阵
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 保存混淆矩阵图像
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()

    # 保存混淆矩阵数值（npy和csv）
    if matrix_path is None:
        matrix_path = os.path.splitext(filename)[0]
    os.makedirs(os.path.dirname(matrix_path), exist_ok=True)

    # 保存为CSV文件和npy文件
    # np.save(f"{matrix_path}.npy", cm)
    # df = pd.DataFrame(cm, index=class_names, columns=class_names)
