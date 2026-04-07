import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 配置中文字体，防止乱码
plt.rcParams['font.sans-serif'] = ['SimSun', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_refined_academic_curves_cn(pth_paths, model_info="EEGNetv4 + VGG-rsscnn",
                                    output_name='smfnet_test_results_cn.png'):
    # 1. 设置 Seaborn 风格
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'SimSun']

    # 2. 定义绘图结构 (2x3 布局)
    plot_config = {
        'all_val_acc': [0, 0, '测试集准确率 (Accuracy, %)', '#1f77b4'],
        'all_val_f1': [0, 1, '测试集 F1 分数 (F1-Score)', '#ff7f0e'],
        'all_val_auc': [0, 2, '测试集 ROC-AUC', '#2ca02c'],
        'all_val_task_loss': [1, 0, '分类任务损失 (Task Loss)', '#d62728'],
        'all_val_common_sim_loss': [1, 1, '特征对齐损失 (Sim Loss)', '#9467bd'],
        'all_val_private_diff_loss': [1, 2, '特征差异损失 (Diff Loss)', '#8c564b']
    }

    # 3. 提取并对齐数据
    all_subject_data = []
    max_epochs = 0

    for p in pth_paths:
        if os.path.exists(p):
            ckpt = torch.load(p, map_location='cpu')
            data = ckpt['history'] if 'history' in ckpt else ckpt
            all_subject_data.append(data)
            max_epochs = max(max_epochs, len(data['all_val_acc']))

    if not all_subject_data:
        print("未找到权重文件，请检查 pth_paths 路径！")
        return

    # 4. 创建 2x3 画布
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    fig.suptitle(f'SMFNet模型测试集性能与收敛过程分析: {model_info}',
                 fontsize=22, fontweight='bold', y=0.98)

    for key, config in plot_config.items():
        row, col, title, color = config
        ax = axes[row, col]

        subject_runs = []
        for sub_history in all_subject_data:
            if key not in sub_history: continue
            run = sub_history[key]
            if len(run) < max_epochs:
                run = list(run) + [run[-1]] * (max_epochs - len(run))
            subject_runs.append(run)

        if not subject_runs: continue

        subject_runs = np.array(subject_runs)
        mean_curve = np.mean(subject_runs, axis=0)
        std_curve = np.std(subject_runs, axis=0)
        epochs = np.arange(1, max_epochs + 1)

        # 绘制均值线和标准差阴影
        ax.plot(epochs, mean_curve, color=color, linewidth=2.5, label='均值 (Mean)')
        ax.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve,
                        color=color, alpha=0.2, label='标准差 ($\pm$Std Dev)')

        # 子图中文修饰
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('迭代轮次 (Epochs)', fontsize=12)
        ax.set_ylabel('数值 (Value)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # ================== 绘制训练阶段划分 ==================
        ax.axvline(x=12, color='gray', linestyle='--', alpha=0.8, linewidth=1.5)
        if max_epochs > 50:
            ax.axvline(x=50, color='gray', linestyle='--', alpha=0.8, linewidth=1.5)

        trans = ax.get_xaxis_transform()

        # 阶段 I (0-12) - 这里将 x 坐标从 6.5 改为了 4.5，让文字往左移
        ax.axvspan(1, 12, color='#e0f7fa', alpha=0.3)
        ax.text(4.5, 0.95, '阶段 I', transform=trans, ha='center', va='top',
                fontsize=11, color='#006064', fontweight='bold')

        # 阶段 II (12-50)
        if max_epochs > 12:
            end_phase2 = min(50, max_epochs)
            ax.axvspan(12, end_phase2, color='#fff9c4', alpha=0.3)
            ax.text((12 + end_phase2) / 2, 0.95, '阶段 II', transform=trans, ha='center', va='top',
                    fontsize=11, color='#f57f17', fontweight='bold')

        # 阶段 III (50+)
        if max_epochs > 50:
            ax.axvspan(50, max_epochs, color='#fce4ec', alpha=0.3)
            ax.text((50 + max_epochs) / 2, 0.95, '阶段 III', transform=trans, ha='center', va='top',
                    fontsize=11, color='#880e4f', fontweight='bold')
        # ==========================================================

        # 坐标轴限制
        if key == 'all_val_acc':
            ax.set_ylim(min(mean_curve) - 5, 102)
        if key in ['all_val_f1', 'all_val_auc']:
            ax.set_ylim(min(mean_curve) - 0.05, 1.02)

        # 添加图例
        ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    pth_files = [f'train_history1/training_history{i}.pth' for i in range(1, 8)]
    plot_refined_academic_curves_cn(pth_files, model_info="EEGNetv1 + PlacesNet-rsscnn")