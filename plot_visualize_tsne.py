import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os
import torch.nn.functional as F

# 导入现有的模块
from MyPP2Dataset import MyPP2Dataset, create_dataloaders
from models.multi_model import MultiModalFusionNetwork
from config import config_multi_model


def extract_features(model, dataloader, device):
    """
    运行前向传播并提取所有样本的解耦特征
    """
    model.eval()

    all_eeg_common = []
    all_image_common = []
    all_eeg_private = []
    all_image_private = []
    all_labels = []

    print("🚀 正在提取特征空间数据，请稍候...")
    with torch.no_grad():
        for left_img, right_img, left_eeg, right_eeg, labels in dataloader:
            left_img, right_img = left_img.to(device), right_img.to(device)
            left_eeg, right_eeg = left_eeg.to(device), right_eeg.to(device)

            # 前向传播提取特征
            _, eeg_common, image_common, eeg_private, image_private = model(
                left_eeg, right_eeg, left_img, right_img
            )

            all_eeg_common.append(eeg_common.cpu().numpy())
            all_image_common.append(image_common.cpu().numpy())
            all_eeg_private.append(eeg_private.cpu().numpy())
            all_image_private.append(image_private.cpu().numpy())
            all_labels.append(labels.numpy())

    # 拼接所有批次
    features = {
        's_E': np.vstack(all_eeg_common),  # eeg_common
        's_I': np.vstack(all_image_common),  # image_common
        'd_E': np.vstack(all_eeg_private),  # eeg_private
        'd_I': np.vstack(all_image_private),  # image_private
        'labels': np.concatenate(all_labels)
    }
    print(f"✅ 特征提取完毕。样本数量: {features['s_E'].shape[0]}, 特征维度: {features['s_E'].shape[1]}")
    return features


def plot_combined_tsne(features, save_path):
    """
    使用 Seaborn 绘制将四个特征置于同一隐空间的全局 t-SNE 可视化图
    """
    sns.set_theme(style="ticks", context="paper", font_scale=1.5)

    # 1. 拼接所有特征，因为现在维度全是 64，可以直接 vstack
    X = np.vstack([features['s_E'], features['s_I'], features['d_E'], features['d_I']])

    n_samples = len(features['s_E'])
    labels = ['$\mathbf{s}_E$ (EEG Common)'] * n_samples + \
             ['$\mathbf{s}_I$ (Image Common)'] * n_samples + \
             ['$\mathbf{d}_E$ (EEG Private)'] * n_samples + \
             ['$\mathbf{d}_I$ (Image Private)'] * n_samples

    print("🧠 正在计算全局 t-SNE 降维 (四个特征共同映射)...")
    # 将所有的特征放入同一个 t-SNE 拟合，保证流形空间一致
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_2d = tsne.fit_transform(X)

    # 构建 Pandas DataFrame 以便于 Seaborn 绘图
    df = pd.DataFrame({
        'TSNE-1': X_2d[:, 0],
        'TSNE-2': X_2d[:, 1],
        'Feature Type': labels
    })

    plt.figure(figsize=(10, 8))

    # 定义颜色字典，确保区分度
    palette = {
        '$\mathbf{s}_E$ (EEG Common)': '#1f77b4',  # 经典蓝
        '$\mathbf{s}_I$ (Image Common)': '#ff7f0e',  # 经典橙
        '$\mathbf{d}_E$ (EEG Private)': '#2ca02c',  # 经典绿
        '$\mathbf{d}_I$ (Image Private)': '#d62728'  # 经典红
    }

    # 增加不同的 Marker 形状，方便黑白打印时依然清晰
    markers = {
        '$\mathbf{s}_E$ (EEG Common)': 'o',  # 圆圈
        '$\mathbf{s}_I$ (Image Common)': 'X',  # 叉号
        '$\mathbf{d}_E$ (EEG Private)': 's',  # 方块
        '$\mathbf{d}_I$ (Image Private)': 'D'  # 菱形
    }

    # 绘制单张散点图
    sns.scatterplot(
        data=df, x='TSNE-1', y='TSNE-2', hue='Feature Type', style='Feature Type',
        markers=markers, palette=palette, alpha=0.7, edgecolor='w', s=80
    )

    plt.title('Global Feature Space Visualization (t-SNE)', fontweight='bold', pad=15)

    # 调整图例位置，避免遮挡数据点
    plt.legend(title='', loc='best', frameon=True, shadow=True, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    # 优化布局并保存
    sns.despine()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"🎉 全局 t-SNE 图表已成功保存至: {save_path}")
    plt.show()
    plt.close()


def plot_feature_orthogonality_heatmap(features, save_path):
    """
    计算并绘制特征空间之间的正交性（绝对余弦相似度）热力图
    注意：维度统一后，去除了之前粗暴的截断，现在是精确计算。
    """
    print("🧠 正在计算特征空间的正交性矩阵...")

    # 转换为 Tensor 以便利用 PyTorch 的矩阵运算
    s_E = torch.tensor(features['s_E'])
    s_I = torch.tensor(features['s_I'])
    d_E = torch.tensor(features['d_E'])
    d_I = torch.tensor(features['d_I'])

    def precise_cosine_similarity(A, B):
        """精准的余弦相似度计算，维度已经是完全匹配的"""
        # 归一化
        A_norm = F.normalize(A, p=2, dim=1)
        B_norm = F.normalize(B, p=2, dim=1)

        # 计算所有样本对的绝对余弦相似度的平均值
        cosine_sim = (A_norm * B_norm).sum(dim=1)
        return cosine_sim.abs().mean().item()

    # 构建 4x4 的相似度矩阵
    feature_list = [
        ('$\mathbf{s}_E$', s_E),
        ('$\mathbf{s}_I$', s_I),
        ('$\mathbf{d}_E$', d_E),
        ('$\mathbf{d}_I$', d_I)
    ]

    n = len(feature_list)
    sim_matrix = np.zeros((n, n))
    labels = [item[0] for item in feature_list]

    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = precise_cosine_similarity(feature_list[i][1], feature_list[j][1])

    # 开始使用 Seaborn 绘图
    sns.set_theme(style="white", context="paper", font_scale=1.5)
    plt.figure(figsize=(8, 6))

    # 使用 Blues 配色，0 附近显示为近乎白色，1 显示为深蓝色，对比鲜明且清爽
    ax = sns.heatmap(
        sim_matrix,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        vmin=0.0, vmax=1.0,
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .8} # 已移除 label
    )

    # 已移除 plt.title

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"🎉 正交性热力图已成功保存至: {save_path}")
    plt.close()


def main():
    config = config_multi_model.Config()
    device = config.device

    dataset = MyPP2Dataset(
        is_flipped=False,
        transform=config.transform,
        subject_id=config.subject_id
    )
    _, test_loader = create_dataloaders(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    # 初始化网络
    model = MultiModalFusionNetwork(
        eeg_model_name=config.base_eeg_model,
        image_model_name=config.base_image_model,
        image_model_type=config.image_model_type,
        subject_id=config.subject_id,
        in_chans=64,
        n_classes=2,
        input_window_samples=2000,
        alpha=config.alpha,
        beta=config.beta,
        ablation_mode="none"
    ).to(device)

    # 确保此处的路径和时间戳是刚训练出来的好结果
    weights_path = f"outputs/outputs_multi_model/{config.base_eeg_model}+{config.base_image_model}_{config.image_model_type}/{config.subject_id}/multimodal_20260406_204841/best_stage3_finetune_backbone.pth"

    if os.path.exists(weights_path):
        print(f"📂 找到权重文件: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ 权重加载成功！")
    else:
        print(f"⚠️ 未找到权重文件: {weights_path}")

    # 提取特征并绘图
    features = extract_features(model, test_loader, device)

    # 1. 全局 t-SNE
    plot_combined_tsne(features, save_path=f"outputs/feature_visualizations/global_tsne_{config.subject_id}.png")

    # 2. 正交性热力图
    plot_feature_orthogonality_heatmap(features,
                                       save_path=f"outputs/feature_visualizations/heatmap_{config.subject_id}.png")


if __name__ == "__main__":
    main()