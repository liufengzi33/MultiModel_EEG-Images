import os
import gc
import torch
import pandas as pd
from torch.utils.data import WeightedRandomSampler, DataLoader

# 引入你的配置、数据和模型类
from config.config_eeg_model import Config
from MyPP2Dataset import MyPP2Dataset, create_dataloaders
from utils.my_transforms import transform_cnn_2
from models.eeg_models import SSBCINet
from train_eegs import EEGTrainer

def main():
    # 1. 定义7个被试的编号
    subjects = ["01gh","02szy","03ysq","04whx","05ly","06wrl","07lxy"]

    # 2. 定义模型及其对应的 weight_decay 专属参数
    model_configs = {
        'EEGNetv1': 1e-3,
        'EEGNetv4': 5e-4,
        'ShallowFBCSPNet': 5e-3
    }

    print(f"🚀 准备开始 EEG 单模态批量训练，总计 {len(subjects) * len(model_configs)} 组实验。")

    all_results = []

    for subject in subjects:
        print(f"\n{'=' * 70}")
        print(f"👤 开始处理被试: {subject}")
        print(f"{'=' * 70}")

        # 新增：专门存储当前被试实验结果的列表
        subject_results = []

        for model_name, wd in model_configs.items():
            print(f"\n🧪 正在运行: 被试={subject} | 模型={model_name} | Weight Decay={wd}")

            # 动态实例化并修改配置
            cfg = Config()
            cfg.subject_id = subject
            cfg.base_model_name = model_name
            cfg.weight_decay = wd

            # ==========================
            # 数据加载与采样器逻辑
            # ==========================
            try:
                dataset = MyPP2Dataset(
                    csv_file="data/safe_qscores_high2low.xlsx",
                    transform=transform_cnn_2,
                    img_dir="data",
                    eeg_dir="data/EEG/seg_eeg_data",
                    is_flipped=False,
                    subject_id=cfg.subject_id
                )

                train_loader, val_loader = create_dataloaders(
                    dataset=dataset,
                    train_ratio=0.8,
                    batch_size=cfg.batch_size,
                    shuffle=True
                )

                # 计算权重并应用 Sampler
                train_labels = [label for _, _, _, _, label in train_loader.dataset]
                train_counts = torch.bincount(torch.tensor(train_labels)).tolist()
                class_weights = 1.0 / torch.tensor(train_counts, dtype=torch.float)
                sample_weights = [class_weights[int(label)] for label in train_labels]

                sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True
                )

                train_loader = DataLoader(
                    dataset=train_loader.dataset,
                    batch_size=cfg.batch_size,
                    sampler=sampler,
                    drop_last=False
                )

                # ==========================
                # 模型初始化与训练
                # ==========================
                model = SSBCINet(
                    base_model_name=cfg.base_model_name,
                    in_chans=64,
                    n_classes=2,
                    input_window_samples=2000
                )

                trainer = EEGTrainer(
                    model,
                    train_loader,
                    val_loader,
                    device=cfg.device,
                    lr=cfg.learning_rate,
                    weight_decay=cfg.weight_decay,
                    patience=cfg.patience,
                    factor=cfg.factor,
                    max_plateaus=cfg.max_lr_plateaus,
                    base_model_name=cfg.base_model_name,
                    subject_id=cfg.subject_id,
                    output_base_dir="outputs/outputs_eeg",
                    pos_weight=None
                )

                # 启动训练
                best_f1, best_acc, best_epoch = trainer.train(epochs=cfg.num_epochs)

                # 提取最佳 Epoch 对应的 AUC
                best_idx = best_epoch - 1
                best_auc = trainer.val_aucs[best_idx]

                result_dict = {
                    'Subject_ID': subject,
                    'Model': model_name,
                    'Weight_Decay': wd,
                    'Best_Epoch': best_epoch,
                    'Val_F1': round(best_f1, 4),
                    'Val_Acc(%)': round(best_acc, 2),
                    'Val_AUC': round(best_auc, 4),
                    'Timestamp': trainer.timestamp,
                    'Save_Dir': trainer.model_output_dir
                }

                # 同时追加到当前被试列表和总列表
                subject_results.append(result_dict)
                all_results.append(result_dict)

            except Exception as e:
                print(f"❌ 运行 {subject} + {model_name} 时发生错误: {e}")
                error_dict = {
                    'Subject_ID': subject, 'Model': model_name, 'Weight_Decay': wd,
                    'Best_Epoch': 'Error', 'Val_F1': 'Error', 'Val_Acc(%)': 'Error',
                    'Val_AUC': 'Error', 'Timestamp': 'Error', 'Save_Dir': 'Error'
                }
                subject_results.append(error_dict)
                all_results.append(error_dict)

            # 强制清理内存与显存
            if 'trainer' in locals(): del trainer
            if 'model' in locals(): del model
            if 'train_loader' in locals(): del train_loader
            if 'val_loader' in locals(): del val_loader
            if 'dataset' in locals(): del dataset
            gc.collect()
            torch.cuda.empty_cache()

        # 核心修改点：当前被试跑完3个模型后，立刻保存一份该被试专属的表格
        subject_df = pd.DataFrame(subject_results)
        os.makedirs('outputs/eegs', exist_ok=True)
        subject_csv_path = f'outputs/eegs/{subject}_eeg_summary.csv'
        subject_df.to_csv(subject_csv_path, index=False, encoding='utf-8-sig')
        print(f"💾 被试 {subject} 的专属统计表已保存至: {subject_csv_path}")

        # 依旧保留全局备份机制，以防万一
        temp_df = pd.DataFrame(all_results)
        temp_df.to_csv('outputs/eegs/eeg_summary_backup.csv', index=False, encoding='utf-8-sig')

        # 3. 保存最终汇总表格 (7个被试，21行数据)
    df = pd.DataFrame(all_results)
    final_csv_path = 'outputs/eegs/7_subjects_eeg_summary.csv'
    df.to_csv(final_csv_path, index=False, encoding='utf-8-sig')

    print(f"\n🎉 全部 7 个被试的单模态 EEG 实验运行完毕！")
    print(f"📊 最终汇总统计表已保存至: {final_csv_path}")
    print("\n📝 最终结果预览:")
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()