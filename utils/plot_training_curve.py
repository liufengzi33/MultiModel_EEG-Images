import os

import seaborn as sns
import matplotlib.pyplot as plt

# 使用 seaborn 的主题风格
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.2)


def live_plot(history, save_path=None, title=None):
    # 将 history 转为 DataFrame 格式，便于使用 seaborn
    epochs = range(1, len(history['train_loss']) + 1)

    # 绘制 Loss 曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制 Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'bo-', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'ro-', label='Val Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    if title:
        plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.85)  # 调整标题位置
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.pause(0.01)
    plt.clf()

