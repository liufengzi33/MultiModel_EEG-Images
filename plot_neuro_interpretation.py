import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import os
from scipy.fft import fft, fftfreq

# 导入你的配置和模型
from models.eeg_models import SSBCINet
from config.config_eeg_model import Config
from models.privileged_model import PrivilegedMultimodalNetwork


def extract_eegnet_features(model_extractor, sfreq=1000):
    """
    提取时域频率响应（带补零插值平滑）和空域绝对激活权重
    """
    conv_temporal = model_extractor.feature_net[2]
    conv_spatial = model_extractor.feature_net[4]

    # ==========================================
    # 1. 提取时域特征 (计算频率响应)
    # ==========================================
    temp_weights = conv_temporal.weight.data.cpu().numpy().squeeze()
    n_filters = temp_weights.shape[0]
    kernel_length = temp_weights.shape[1]

    pad_length = 2000  # 补零到 2000 提升 FFT 分辨率
    freqs = fftfreq(pad_length, 1 / sfreq)
    positive_freqs_idx = (freqs >= 1) & (freqs <= 45)  # 截取 1~45 Hz
    valid_freqs = freqs[positive_freqs_idx]

    freq_responses = []
    for i in range(n_filters):
        padded_filter = np.pad(temp_weights[i], (0, pad_length - kernel_length), 'constant')
        freq_responses.append(np.abs(fft(padded_filter))[positive_freqs_idx])

    mean_freq_response = np.mean(freq_responses, axis=0)
    # 频率响应各自归一化，以便观察波峰
    mean_freq_response = (mean_freq_response - np.min(mean_freq_response)) / (
                np.max(mean_freq_response) - np.min(mean_freq_response))

    # ==========================================
    # 2. 提取空域特征 (用于 Topomap)
    # ==========================================
    spatial_weights = conv_spatial.weight.data.cpu().numpy().squeeze()
    # 取绝对值的均值，代表该通道的激活强度
    channel_importance = np.mean(np.abs(spatial_weights), axis=0)

    return valid_freqs, mean_freq_response, channel_importance


def plot_real_neuro_interpretation_final(base_freqs, base_psd, base_spatial,
                                         dist_freqs, dist_psd, dist_spatial, save_path):
    """
    终极发表版：
    1. 独立展现空间拓扑结构，Colorbar 位于地形图左侧。
    2. 频率响应图标注具体的频段范围。
    3. 所有子图的 Title 统一放在图表下方。
    """
    # 准备 64 通道信息
    ch_names = ['Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8', 'Fz', 'F1', 'F2', 'F3', 'F4',
                'F5', 'F6', 'F7', 'F8', 'FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7',
                'FT8', 'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'T7', 'T8', 'CP1', 'CP2',
                'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8', 'Pz', 'P3', 'P4', 'P5', 'P6', 'P7',
                'P8', 'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'Oz', 'O1', 'O2',
                'ECG', 'HEOR', 'HEOL', 'VEOU', 'VEOL']
    ch_types = ['eeg'] * 59 + ['ecg'] + ['eog'] * 4
    info = mne.create_info(ch_names, sfreq=1000, ch_types=ch_types)
    info.set_montage(mne.channels.make_standard_montage('standard_1005'))
    eeg_indices = mne.pick_types(info, eeg=True)
    info_eeg = mne.pick_info(info, eeg_indices)

    # 截取前 59 个纯 EEG 权重
    b_sp = base_spatial[:len(eeg_indices)]
    d_sp = dist_spatial[:len(eeg_indices)]

    sns.set_theme(style="ticks", context="paper", font_scale=1.4)
    # 高度稍作增加以容纳底部的 Title
    fig = plt.figure(figsize=(18, 9.5))

    # 布局优化：增加 hspace (0.35 -> 0.45) 防止上下两行的文字/图表打架
    gs = fig.add_gridspec(2, 4, width_ratios=[0.08, 1.2, 0.2, 2.0], hspace=0.45, wspace=0.1)

    cmap_shared = "magma"

    # --- (a) Baseline Topomap ---
    ax_a = fig.add_subplot(gs[0, 1])
    im_a, _ = mne.viz.plot_topomap(b_sp, info_eeg, axes=ax_a, cmap=cmap_shared, show=False)
    # 使用 y=-0.15 将标题下移
    ax_a.set_title("(a) Baseline Spatial Activation", fontweight='bold', y=-0.15)

    # Baseline 独立 Colorbar
    ax_cb_a = fig.add_subplot(gs[0, 0])
    cb_a = fig.colorbar(im_a, cax=ax_cb_a, orientation='vertical')
    cb_a.set_label("Weight", fontweight='bold')
    ax_cb_a.yaxis.set_ticks_position('left')
    ax_cb_a.yaxis.set_label_position('left')

    # --- (c) Distilled Topomap ---
    ax_c = fig.add_subplot(gs[1, 1])
    im_c, _ = mne.viz.plot_topomap(d_sp, info_eeg, axes=ax_c, cmap=cmap_shared, show=False)
    # 使用 y=-0.15 将标题下移
    ax_c.set_title("(c) Online KD Spatial Activation", fontweight='bold', y=-0.15)

    # Distilled 独立 Colorbar
    ax_cb_c = fig.add_subplot(gs[1, 0])
    cb_c = fig.colorbar(im_c, cax=ax_cb_c, orientation='vertical')
    cb_c.set_label("Weight", fontweight='bold')
    ax_cb_c.yaxis.set_ticks_position('left')
    ax_cb_c.yaxis.set_label_position('left')

    # --- (b) & (d) Frequency Response ---
    for i, (f, p, title, color, ax_idx) in enumerate([
        (base_freqs, base_psd, "(b) Baseline Filter Frequency Response", "#4C72B0", gs[0, 3]),
        (dist_freqs, dist_psd, "(d) Distilled Filter Frequency Response", "#C44E52", gs[1, 3])
    ]):
        ax = fig.add_subplot(ax_idx)
        ax.plot(f, p, color=color, lw=3)
        ax.fill_between(f, p, alpha=0.3, color=color)

        # 统一显示三波段阴影
        bands = {
            'Alpha\n(8-12 Hz)': (8, 12, '#7F8C8D'),
            'Beta\n(13-30 Hz)': (13, 30, '#E67E22'),
            'Gamma\n(>30 Hz)': (30, 45, '#F1C40F')
        }
        for name, (l, h, c) in bands.items():
            ax.axvspan(l, h, color='gray', alpha=0.1)
            ax.text((l + h) / 2, 0.88, name, ha='center', va='top', color=c, fontweight='bold', fontsize=12)

        ax.set_xlim(1, 45)
        ax.set_ylim(0, 1.1)
        sns.despine(ax=ax)

        # 核心修改：动态调整 Title 位置
        if i == 0:  # 图 (b)
            ax.set_xticks([])  # 隐藏图 b 的横坐标轴
            ax.set_title(title, fontweight='bold', y=-0.15)  # 没有 xlabel，-0.15 即可
        else:  # 图 (d)
            ax.set_xlabel("Frequency (Hz)", fontweight='bold')
            ax.set_title(title, fontweight='bold', y=-0.32)  # 有 xlabel，需推得更低 (-0.32) 避开 label

    # 留出底部边距，防止图 d 的 title 导出时被截断
    plt.subplots_adjust(bottom=0.15)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"🎉 终极学术排版图表已保存至: {save_path}")
    plt.show()
    plt.close()  # 生成后关闭，清理内存


def main():
    config = Config()
    device = torch.device('cpu')

    # ==========================================
    # 1. 初始化模型实例
    # ==========================================
    print("⚙️ 初始化网络架构...")
    baseline_model = SSBCINet(base_model_name=config.base_model_name, in_chans=64, n_classes=2)

    full_kd_network = PrivilegedMultimodalNetwork(
        student_modality="eeg", eeg_model_name=config.base_model_name,
        subject_id=config.subject_id, use_pretrained_eeg=False, use_pretrained_image=False, feature_dim=64
    )
    distilled_student_model = full_kd_network.student_network

    # ==========================================
    # 2. 完美的权重加载逻辑 (直接使用你的代码)
    # ==========================================
    baseline_path = f"outputs/outputs_eeg/{config.base_model_name}/{config.subject_id}/best_model.pth"
    distilled_path = f"outputs/outputs_privileged/{config.base_model_name}_VGG_sscnn_student_eeg/{config.subject_id}/20260411_142711/distilled_student_only.pth"

    # 加载基线模型权重
    if os.path.exists(baseline_path):
        baseline_checkpoint = torch.load(baseline_path, map_location=device)
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

        # 2. 清理可能的包装前缀
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
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
    # 3. 提取特征
    # ==========================================
    print("🧠 正在解剖基线模型权重...")
    base_freqs, base_psd, base_spatial = extract_eegnet_features(baseline_model.feature_extractor)

    print("🧠 正在解剖蒸馏模型权重...")
    # 提取特权网络中的 student_network 里面的 feature_extractor
    dist_freqs, dist_psd, dist_spatial = extract_eegnet_features(
        distilled_student_model['feature_path'].feature_extractor)

    # ==========================================
    # 🕵️ 新增：强力断言与差异诊断
    # ==========================================
    print("\n📊 --- 正在比对底层提取特征 ---")

    # 1. 频率轴 (Freqs)
    # 这个必然是 True，因为这是由采样率和补零长度算出来的横坐标，与权重无关。
    is_freqs_same = np.allclose(base_freqs, dist_freqs)
    print(f"[*] 频率横轴 (Freqs) 是否一致 : {is_freqs_same}")

    # 2. 时域频率响应 (PSD)
    # 代表 conv_temporal 的权重是否有差异
    is_psd_same = np.allclose(base_psd, dist_psd)
    print(f"[*] 时域频率响应 (PSD) 是否一致 : {is_psd_same}")

    # 3. 空域激活权重 (Spatial)
    # 代表 conv_spatial 的权重是否有差异
    is_spatial_same = np.allclose(base_spatial, dist_spatial)
    print(f"[*] 空域激活权重 (Spatial) 是否一致: {is_spatial_same}")

    print("----------------------------------")
    if is_psd_same and is_spatial_same:
        print("🚨🚨 致命诊断结果 🚨🚨")
        print("基线和蒸馏模型的底层时空卷积核【完全一模一样】！")
        print("-> 原因 1：在三阶段训练时，你的学生网络底层被冻结了（requires_grad=False）。")
        print("-> 原因 2：优化器（Optimizer）初始化时，没有包含这几个底层的参数。")
        print("-> 结论：画出来的图 A 和 C、B 和 D 必定完全相同。必须回去修改训练代码并重新跑蒸馏！")
    elif is_psd_same and not is_spatial_same:
        print("⚠️ 局部一致诊断结果")
        print("-> 只有空域权重改变了，时域权重没变。这说明蒸馏过程只影响了 Spatial 层。")
    elif not is_psd_same and is_spatial_same:
        print("⚠️ 局部一致诊断结果")
        print("-> 只有时域权重改变了，空域权重没变。")
    else:
        print("✅ 完美通过：")
        print("-> 基线和蒸馏的特征存在实质性差异，知识蒸馏成功渗透并重塑了底层的所有权重！可以放心画图！")
    print("==========================================\n")

    # 4. 防呆校验 (可选，只为图个安心)
    if np.allclose(base_spatial, dist_spatial):
        print("⚠️ 注意：提取出的空间权重完全一致。如果图 A 和图 C 仍没有区别，需检查蒸馏过程是否意外冻结了该层卷积。")
    else:
        print("✅ 检测到权重存在实质性差异，准备生成热力对比图！")

    # ==========================================
    # 4. 绘图
    # ==========================================
    os.makedirs("outputs/feature_visualizations", exist_ok=True)
    plot_real_neuro_interpretation_final(
        base_freqs, base_psd, base_spatial,
        dist_freqs, dist_psd, dist_spatial,
        save_path=f"outputs/feature_visualizations/neuro_interpretation_{config.subject_id}_Final.png"
    )


if __name__ == "__main__":
    main()