import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import os

# 导入基线模型和配置
from models.eeg_models import SSBCINet
from config.config_eeg_model import Config
from MyPP2Dataset import MyPP2Dataset, create_dataloaders_by_order
from models.privileged_model import PrivilegedMultimodalNetwork


def extract_baseline_features(model, dataloader, device):
    """提取基线纯 EEG 模型 (SSBCINet) 的 512 维特征"""
    model.eval()
    model.to(device)

    features, labels = [], []
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach().cpu().numpy()

        return hook

    # 基线模型挂载在 fusion 上 (输出维度为 512)
    hook_handle = model.fusion.register_forward_hook(get_activation('features'))

    with torch.no_grad():
        for left_img, right_img, left_eeg, right_eeg, label in dataloader:
            left_eeg, right_eeg = left_eeg.to(device), right_eeg.to(device)
            _ = model(left_eeg, right_eeg)

            features.append(activation['features'])
            labels.append(label.cpu().numpy())

    hook_handle.remove()
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def extract_distilled_features(student_network, dataloader, device):
    """提取蒸馏后学生网络的 64 维特征 (已被图像重塑的空间)"""
    student_network.eval()
    student_network.to(device)

    features, labels = [], []
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach().cpu().numpy()

        return hook

    # 学生网络挂载在 feature_encoder 上 (输出维度为 64)
    hook_handle = student_network['feature_encoder'].register_forward_hook(get_activation('features'))

    with torch.no_grad():
        for left_img, right_img, left_eeg, right_eeg, label in dataloader:
            left_eeg, right_eeg = left_eeg.to(device), right_eeg.to(device)

            # 手动执行学生网络的前向传播 (匹配 ModuleDict 结构)
            base_features = student_network['feature_path'](left_eeg, right_eeg)
            _ = student_network['feature_encoder'](base_features)  # 触发 hook

            features.append(activation['features'])
            labels.append(label.cpu().numpy())

    hook_handle.remove()
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def plot_tsne_comparison(features_baseline, features_distilled, labels, save_path):
    """绘制并排的流形对比图 (期刊风格)"""
    print("🧠 正在计算纯 EEG 模型的 t-SNE (Baseline)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=40, init='pca', learning_rate='auto')
    tsne_baseline = tsne.fit_transform(features_baseline)

    print("🧠 正在计算蒸馏后 EEG 模型的 t-SNE (Online KD)...")
    tsne_distilled = tsne.fit_transform(features_distilled)

    sns.set_theme(style="ticks", context="paper", font_scale=1.5)
    # 高度稍微增加一点点，以便给底部的文字留出空间
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    palette = {0: "#4C72B0", 1: "#C44E52"}
    markers = {0: 'o', 1: 's'}
    class_names = ["Left Safer (Class 0)", "Right Safer (Class 1)"]

    # ================= (a) Baseline =================
    sns.scatterplot(
        x=tsne_baseline[:, 0], y=tsne_baseline[:, 1],
        hue=labels, style=labels, markers=markers, palette=palette,
        alpha=0.8, s=60, ax=axes[0], legend=False, edgecolor="w", linewidth=0.5
    )
    # 1. 命名坐标轴
    axes[0].set_xlabel("t-SNE Dimension 1", fontsize=14, fontweight='bold', labelpad=10)
    axes[0].set_ylabel("t-SNE Dimension 2", fontsize=14, fontweight='bold', labelpad=10)
    # 隐藏刻度值，保持图面整洁
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    # 2. 将 title 移至下方 (利用负向的 y 值)
    axes[0].set_title("(a) Pure EEG Model (w/o KD)", fontsize=16, fontweight='bold', y=-0.22)

    # ================= (b) Distilled =================
    sns.scatterplot(
        x=tsne_distilled[:, 0], y=tsne_distilled[:, 1],
        hue=labels, style=labels, markers=markers, palette=palette,
        alpha=0.8, s=60, ax=axes[1], edgecolor="w", linewidth=0.5
    )
    # 1. 命名坐标轴
    axes[1].set_xlabel("t-SNE Dimension 1", fontsize=14, fontweight='bold', labelpad=10)
    axes[1].set_ylabel("t-SNE Dimension 2", fontsize=14, fontweight='bold', labelpad=10)
    # 隐藏刻度值
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    # 2. 将 title 移至下方
    axes[1].set_title("(b) Distilled EEG Model (Online KD)", fontsize=16, fontweight='bold', y=-0.22)

    # 图例设置
    handles = [plt.Line2D([0], [0], marker=markers[i], color='w', markerfacecolor=palette[i], markersize=10) for i in
               [0, 1]]
    axes[1].legend(handles, class_names, title="Decision", loc='best', fontsize=12, frameon=True, shadow=True)

    sns.despine()
    # 适度留白底部以防止下方 title 被截断
    plt.subplots_adjust(bottom=0.2)
    # bbox_inches='tight' 会确保图片保存时包含外部所有文本（包括被移到底部的title）
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"🎉 图像已成功保存至: {save_path}")

    plt.show()
    # 建议保存完就关闭，避免占内存
    plt.close()


def main():
    config = Config()
    device = torch.device(config.device)
    print(f"Using device: {device}")

    # ==========================================
    # 1. 准备数据
    # ==========================================
    print(f"📂 Loading real EEG data for subject {config.subject_id}...")
    dataset = MyPP2Dataset(is_flipped=False, transform=config.transform, subject_id=config.subject_id)

    # 提取评估数据
    # _, test_loader = create_dataloaders_by_order(dataset, batch_size=config.batch_size, shuffle=False)

    # --- 修改后 ---
    from torch.utils.data import DataLoader

    print(f"📂 Loading ALL {len(dataset)} real EEG data for subject {config.subject_id}...")
    # 直接对整个 dataset 创建 dataloader，这样就包含全部 300 个样本了
    test_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)


    # ==========================================
    # 2. 初始化网络架构
    # ==========================================
    print("⚙️ Initializing networks...")
    # 基线网络 (SSBCINet)
    baseline_model = SSBCINet(
        base_model_name=config.base_model_name,
        in_chans=64, n_classes=2, input_window_samples=2000
    )

    # 蒸馏后的学生网络 (从 PrivilegedMultimodalNetwork 提取)
    full_kd_network = PrivilegedMultimodalNetwork(
        student_modality="eeg",
        eeg_model_name=config.base_model_name,
        subject_id=config.subject_id,
        use_pretrained_eeg=False,
        use_pretrained_image=False,
        feature_dim=64  # 确保这里的维度和你训练时一致
    )
    distilled_student_model = full_kd_network.student_network

    # ==========================================
    # 3. 安全加载权重 (解决格式不匹配问题)
    # ==========================================
    baseline_path = f"outputs/outputs_eeg/{config.base_model_name}/{config.subject_id}/best_model.pth"
    distilled_path = f"outputs/outputs_privileged/{config.base_model_name}_VGG_sscnn_student_eeg/{config.subject_id}/20260410_145333/distilled_student_only.pth"

    # 加载基线模型权重
    if os.path.exists(baseline_path):
        baseline_checkpoint = torch.load(baseline_path, map_location=device)
        # 兼容保存的是字典还是纯参数
        if 'model_state_dict' in baseline_checkpoint:
            baseline_model.load_state_dict(baseline_checkpoint['model_state_dict'])
        else:
            baseline_model.load_state_dict(baseline_checkpoint)
        print(f"✅ Loaded baseline weights from: {baseline_path}")
    else:
        print(f"⚠️ Baseline weights not found at {baseline_path}")

    # 加载蒸馏模型权重
    if os.path.exists(distilled_path):
        print("📂 找到文件！正在安全提取真实权重...")
        distilled_checkpoint = torch.load(distilled_path, map_location=device)

        # 1. 拨开外层壳，拿到真正的字典
        if 'student_state_dict' in distilled_checkpoint:
            state_dict = distilled_checkpoint['student_state_dict']
        elif 'model_state_dict' in distilled_checkpoint:
            state_dict = distilled_checkpoint['model_state_dict']
        else:
            state_dict = distilled_checkpoint

        # 2. 清理可能的包装前缀，不做任何乱七八糟的后缀猜测
        clean_state_dict = {}
        for k, v in state_dict.items():
            # 如果保存时使用了 DataParallel，会自动多出 'module.'，去掉它
            if k.startswith('module.'):
                k = k[7:]
            # 如果保存时是整个网络保存的，多出了 'student_network.'，去掉它
            if k.startswith('student_network.'):
                k = k[16:]

            clean_state_dict[k] = v

        # 3. 严格加载
        try:
            distilled_student_model.load_state_dict(clean_state_dict, strict=True)
            print(f"✅ 蒸馏模型权重完美加载！原汁原味地还原了模型参数。")
        except Exception as e:
            print(f"❌ 权重加载失败！报错信息: {e}")
            print("\n🚨 让我们看看真实的字典里到底保存了什么名字：")
            print(list(clean_state_dict.keys())[:10])
    else:
        print(f"⚠️ 严重警告：在硬盘上根本找不到这个文件！")
    print("==========================================\n")

    # ==========================================
    # 4. 提取特征并绘图
    # ==========================================
    print("\n🚀 Extracting Baseline Features...")
    features_baseline, labels_baseline = extract_baseline_features(baseline_model, test_loader, device)

    print("🚀 Extracting Distilled Features...")
    features_distilled, labels_distilled = extract_distilled_features(distilled_student_model, test_loader, device)

    os.makedirs("outputs/feature_visualizations", exist_ok=True)
    save_img_path = f"outputs/feature_visualizations/tsne_manifold_{config.subject_id}.png"

    plot_tsne_comparison(features_baseline, features_distilled, labels_baseline, save_path=save_img_path)


if __name__ == "__main__":
    main()