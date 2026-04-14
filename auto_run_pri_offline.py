import os
import glob
import json
import numpy as np
import pandas as pd
from datetime import datetime
import torch

from config.config_privileged_model import get_config
from train_offline_distillation import OfflineDistillationTrainer


def find_teacher_ckpt(base_path, eeg_model, img_model, img_type, subject, mode):
    """根据模式动态寻找最新的教师模型权重"""
    dir_name = f"{eeg_model}+{img_model}_{img_type}"

    if mode == "Intra-subject":
        search_dir = os.path.join(base_path, 'outputs_multi_model', dir_name, subject)
    elif mode == "Cross-subject":
        search_dir = os.path.join(base_path, 'outputs_multi_model', dir_name, f"LOSO_test_{subject}")
    else:
        raise ValueError("不支持的模式")

    if not os.path.exists(search_dir):
        raise FileNotFoundError(f"❌ 找不到教师模型目录: {search_dir}")

    subdirs = [os.path.join(search_dir, d) for d in os.listdir(search_dir) if
               os.path.isdir(os.path.join(search_dir, d))]
    if not subdirs:
        raise FileNotFoundError(f"❌ 目录为空: {search_dir}")

    # 获取所有时间戳文件夹，找到最老（最早）的那个
    oldest_dir = min(subdirs, key=os.path.getctime)
    ckpt_path = os.path.join(oldest_dir, "best_stage3_finetune_backbone.pth")
    if not os.path.exists(ckpt_path):
        pt_files = glob.glob(os.path.join(oldest_dir, "*.pth"))
        ckpt_path = max(pt_files, key=os.path.getctime) if pt_files else None

    return ckpt_path, oldest_dir


def main():
    # ================= 配置区域 =================
    all_subjects = ["01gh", "02szy", "03ysq", "04whx", "05ly", "06wrl", "07lxy"]
    MODE = "Intra-subject"  # 可选: "Intra-subject" 或 "Cross-subject"

    # 统计数据根目录 (CSV输出位置)
    stats_root = os.path.join("outputs/offline_pri", MODE)
    os.makedirs(stats_root, exist_ok=True)

    # 权重与图表根目录 (Trainer输出位置)
    weights_root = os.path.join("outputs", "outputs_offline_distill", MODE)
    # ============================================

    all_results = []

    print(f"🚀 启动批量离线蒸馏 | 模式: {MODE}")

    for subject in all_subjects:
        print(f"\n{'=' * 60}\n正在处理被试: {subject}\n{'=' * 60}")

        config = get_config('default')
        config.subject_id = subject
        config.train_mode = MODE
        config.all_subjects = all_subjects

        # 获取模型配置名称：EEGNetv4_VGG_sscnn_student_eeg
        model_subdir = f"{config.eeg_model_name}_{config.image_model_name}_{config.image_model_type}_student_{config.student_modality}"

        # 1. 寻找教师
        try:
            teacher_ckpt, teacher_dir = find_teacher_ckpt(
                config.base_path, config.eeg_model_name,
                config.image_model_name, config.image_model_type,
                subject, MODE
            )
            config.teacher_ckpt_path = teacher_ckpt
        except Exception as e:
            print(f"⚠️ 跳过被试 {subject}: {e}")
            continue

        # 2. 设置 Trainer 的输出目录 (权重、pth、Loss图)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 目标：outputs/outputs_offline_distill/Intra-subject/EEGNetv4_VGG_.../01gh/timestamp
        final_out_dir = os.path.join(weights_root, model_subdir, subject, timestamp)
        os.makedirs(final_out_dir, exist_ok=True)

        config.output_dir = final_out_dir
        # 我们在这里加一个标记，告诉 Trainer 路径已经死锁了
        config.is_path_locked = True

        # 3. 执行训练
        trainer = OfflineDistillationTrainer(config)
        trainer.train()

        # 4. 提取指标并保存单被试 CSV
        history_path = os.path.join(final_out_dir, 'offline_history.pth')
        if os.path.exists(history_path):
            history = torch.load(history_path)
            best_idx = np.argmax(history['val_student_acc'])

            # 计算教师准确率
            teacher_history_path = os.path.join(teacher_dir, 'training_history.pth')
            teacher_acc = 100.0
            if os.path.exists(teacher_history_path):
                t_hist = torch.load(teacher_history_path)
                if 'all_val_acc' in t_hist: teacher_acc = max(t_hist['all_val_acc'])

            best_acc = history['val_student_acc'][best_idx]
            krr = (best_acc / teacher_acc) * 100 if teacher_acc > 0 else 0.0

            res = {
                "Subject": subject,
                "Acc (%)": best_acc,
                "F1 Score": history['val_student_f1'][best_idx],
                "AUC-ROC": history['val_student_auc'][best_idx],
                "Teacher Acc (%)": teacher_acc,
                "KRR (%)": krr
            }
            all_results.append(res)

            # --- 精确要求的输出 ---
            # 统计 CSV 路径：offline_pri/Cross-subject/01gh_offline_pri_distill.csv
            subject_csv_path = os.path.join(stats_root, f"{subject}_offline_pri_distill.csv")
            pd.DataFrame([res]).to_csv(subject_csv_path, index=False)
            print(f"📊 统计已保存至: {subject_csv_path}")

    # 5. 汇总统计
    if all_results:
        df = pd.DataFrame(all_results)
        summary_path = os.path.join(stats_root, f"TOTAL_summary_{MODE}.csv")
        df.to_csv(summary_path, index=False)
        print(f"\n✅ 汇总完成！全表保存在: {summary_path}")


if __name__ == "__main__":
    main()