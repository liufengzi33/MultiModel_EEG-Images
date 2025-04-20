import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 使用 seaborn 的主题风格
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.2)

def plot_training_curve(history, title):
    # 将 history 转为 DataFrame 格式，便于使用 seaborn
    df = pd.DataFrame({
        'Epoch': range(1, len(history['train_loss']) + 1),
        'Train Loss': history['train_loss'],
        'Val Loss': history['val_loss'],
        'Train Acc': history['train_acc'],
        'Val Acc': history['val_acc'],
    })

    plt.figure(figsize=(12, 4))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    sns.lineplot(x='Epoch', y='value', hue='variable',
                 data=pd.melt(df[['Epoch', 'Train Loss', 'Val Loss']], ['Epoch']),
                 linewidth=2.0)
    plt.title('Loss Curve')
    plt.ylabel('Loss')

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    sns.lineplot(x='Epoch', y='value', hue='variable',
                 data=pd.melt(df[['Epoch', 'Train Acc', 'Val Acc']], ['Epoch']),
                 linewidth=2.0)
    plt.title('Accuracy Curve')
    plt.ylabel('Accuracy (%)')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
