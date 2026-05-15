import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def plot_loss_curves(pth_file_path, save_dir="./"):
    """
    读取训练历史 pth 文件，并绘制 2x2 的损失大图保存
    图表中所有字体均强制设置为 30pt
    """
    if not os.path.exists(pth_file_path):
        print(f"❌ 找不到文件: {pth_file_path}")
        return

    # 加载历史文件数据
    print(f"正在加载训练历史: {pth_file_path}")
    try:
        history = torch.load(pth_file_path, map_location='cpu')
    except Exception as e:
        print(f"❌ 加载文件失败: {e}")
        return

    # 确定 Epoch 的数量
    if 'all_train_loss' in history:
        epochs = range(1, len(history['all_train_loss']) + 1)
    else:
        print("❌ 未在文件中找到 'all_train_loss' 键，无法确定 Epoch 数量。请检查字典键名。")
        return

    print("开始绘制图表...")

    # 设置基础风格
    sns.set_theme(style="whitegrid", context="paper")

    # 【核心修改区】：全局强制替换所有相关的字体大小为 30pt
    plt.rcParams.update({
        'font.size': 30,  # 全局默认字体大小
        'axes.titlesize': 30,  # 子图标题字体大小
        'axes.labelsize': 30,  # x轴和y轴标签字体大小
        'xtick.labelsize': 30,  # x轴刻度数字字体大小
        'ytick.labelsize': 30,  # y轴刻度数字字体大小
        'legend.fontsize': 30,  # 图例字体大小
        'figure.titlesize': 30  # 整体大标题字体大小
    })

    # ==========================================
    # 图 1: 损失大图 (包含总损失和所有子损失, 2x2 排版)
    # 因为字体变成了30pt，画布尺寸从(16,10)放大到(24,16)以防止文字严重重叠
    # 同时将 linewidth 调为 4，markersize 调为 10，保持画面视觉平衡
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(24, 16))

    # 1. 总损失
    axes[0, 0].plot(epochs, history.get('all_train_loss', []), linestyle='-', marker='o', markersize=10,
                    label='Train Loss', linewidth=4, alpha=0.8)
    axes[0, 0].plot(epochs, history.get('all_val_loss', []), linestyle='-', marker='s', markersize=10,
                    label='Val Loss', linewidth=4, alpha=0.8)
    axes[0, 0].set_title('Total Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch', fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontweight='bold')
    axes[0, 0].legend(loc='upper right', frameon=True, shadow=True)

    # 2. 任务损失
    axes[0, 1].plot(epochs, history.get('all_task_loss', []), linestyle='-', marker='o', markersize=10,
                    label='Train Task Loss', linewidth=4, alpha=0.8, color='crimson')
    axes[0, 1].plot(epochs, history.get('all_val_task_loss', []), linestyle='-', marker='s', markersize=10,
                    label='Val Task Loss', linewidth=4, alpha=0.8, color='coral')
    axes[0, 1].set_title('Task Loss', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch', fontweight='bold')
    axes[0, 1].set_ylabel('Loss', fontweight='bold')
    axes[0, 1].legend(loc='upper right', frameon=True, shadow=True)

    # 3. 共同相似度损失
    axes[1, 0].plot(epochs, history.get('all_common_sim_loss', []), linestyle='-', marker='o', markersize=10,
                    label='Train Common Sim Loss', linewidth=4, alpha=0.8, color='forestgreen')
    axes[1, 0].plot(epochs, history.get('all_val_common_sim_loss', []), linestyle='-', marker='s', markersize=10,
                    label='Val Common Sim Loss', linewidth=4, alpha=0.8, color='limegreen')
    axes[1, 0].set_title('Common Similarity Loss', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch', fontweight='bold')
    axes[1, 0].set_ylabel('Loss', fontweight='bold')
    axes[1, 0].legend(loc='upper right', frameon=True, shadow=True)

    # 4. 私有差异损失
    axes[1, 1].plot(epochs, history.get('all_private_diff_loss', []), linestyle='-', marker='o', markersize=10,
                    label='Train Private Diff Loss', linewidth=4, alpha=0.8, color='purple')
    axes[1, 1].plot(epochs, history.get('all_val_private_diff_loss', []), linestyle='-', marker='s', markersize=10,
                    label='Val Private Diff Loss', linewidth=4, alpha=0.8, color='violet')
    axes[1, 1].set_title('Private Difference Loss', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch', fontweight='bold')
    axes[1, 1].set_ylabel('Loss', fontweight='bold')
    axes[1, 1].legend(loc='upper right', frameon=True, shadow=True)

    # 布局调整与保存
    plt.tight_layout()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, 'all_losses_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 恢复默认风格与字体设置，防止影响同一个环境下的后续其他图表绘制
    sns.reset_orig()
    plt.rcParams.update(plt.rcParamsDefault)
    print(f"✅ 图表已保存至: {save_path}")

if __name__ == "__main__":
    # 指定你上传的 .pth 文件的相对或绝对路径
    HISTORY_FILE = "D:\PycharmProject\MultiModel_EEG&Image\outputs\multimodal_20260406_213848\\training_history.pth"

    # 运行绘图函数
    plot_loss_curves(HISTORY_FILE)