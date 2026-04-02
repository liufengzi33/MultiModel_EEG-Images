import os
import gc
import torch
import pandas as pd
import itertools
from config.config_multi_model import Config
from train_multimodal_three_phase import MultiStageTrainer


def main():
    # =====================================================================
    # 🎯 手动指定消融模式 (每次运行前在这里修改!)
    # 可选值: 'baseline_concat', 'no_cmd', 'no_ortho', 'none'
    # =====================================================================
    current_ablation_mode = 'baseline_concat'

    # 动态创建专属输出目录，例如: outputs/multi/baseline_concat
    out_dir = f'outputs/multi/{current_ablation_mode}'
    os.makedirs(out_dir, exist_ok=True)

    subjects = ["01gh", "02szy", "03ysq", "04whx", "05ly", "06wrl", "07lxy"]

    # 缩小模型测试范围
    eeg_models = ['EEGNetv1', 'EEGNetv4', 'ShallowFBCSPNet']
    image_models = ['AlexNet', 'PlacesNet', 'VGG']
    image_types = ['sscnn', 'rsscnn']

    # 生成组合：EEG x Image x Type (不再将 ablation_mode 放进去相乘)
    combinations = list(itertools.product(eeg_models, image_models, image_types))

    print(f"🚀 准备开始【{current_ablation_mode}】模式的自动化运行，总计 {len(subjects) * len(combinations)} 组实验组合。")
    print(f"📁 结果将统一保存在: {out_dir}\n")

    all_results = []

    for subject_id in subjects:
        print(f"\n{'=' * 80}")
        print(f"👤 开始处理被试: {subject_id}")
        print(f"{'=' * 80}")

        subject_results = []

        # 注意这里的解包：现在只有三个参数
        for i, (eeg, img, img_type) in enumerate(combinations):
            print(f"\n{'=' * 70}")
            print(
                f"🧪 实验 {i + 1}/{len(combinations)}: Sub={subject_id} | EEG={eeg} | Img={img} | Type={img_type} | Ablation={current_ablation_mode}")
            print(f"{'=' * 70}")

            config = Config()
            config.subject_id = subject_id
            config.base_eeg_model = eeg
            config.base_image_model = img
            config.image_model_type = img_type
            config.ablation_mode = current_ablation_mode  # 动态注入你手动设定的消融模式

            trainer = MultiStageTrainer(config)
            trainer.setup_data()
            trainer.setup_model()

            try:
                model, history = trainer.train()

                val_losses = history['all_val_loss']
                best_epoch_idx = val_losses.index(min(val_losses))

                best_acc = history['all_val_acc'][best_epoch_idx]
                best_f1 = history['all_val_f1'][best_epoch_idx]
                best_auc = history['all_val_auc'][best_epoch_idx]

                timestamp = trainer.save_dir.split('multimodal_')[-1]

                result_dict = {
                    'Subject_ID': subject_id,
                    'Ablation': current_ablation_mode,  # 记录消融条件
                    'EEG_Model': eeg,
                    'Image_Model': img,
                    'Image_Type': img_type,
                    'Best_Epoch': best_epoch_idx + 1,
                    'Val_Acc(%)': round(best_acc, 2),
                    'Val_F1': round(best_f1, 4),
                    'Val_AUC': round(best_auc, 4),
                    'Timestamp': timestamp
                }

                subject_results.append(result_dict)
                all_results.append(result_dict)

            except Exception as e:
                print(f"❌ 运行组合发生错误: {e}")
                error_dict = {
                    'Subject_ID': subject_id, 'Ablation': current_ablation_mode, 'EEG_Model': eeg, 'Image_Model': img,
                    'Image_Type': img_type,
                    'Best_Epoch': 'Error', 'Val_Acc(%)': 'Error', 'Val_F1': 'Error', 'Val_AUC': 'Error',
                    'Timestamp': 'Error'
                }
                subject_results.append(error_dict)
                all_results.append(error_dict)

            del trainer
            if 'model' in locals():
                del model
            gc.collect()
            torch.cuda.empty_cache()

        # 实时保存当前被试的汇总表到专属目录
        df_subj = pd.DataFrame(subject_results)
        csv_subj_path = f'{out_dir}/{subject_id}_ablation_summary.csv'
        df_subj.to_csv(csv_subj_path, index=False, encoding='utf-8-sig')

        # 实时更新全局备份文件到专属目录
        pd.DataFrame(all_results).to_csv(f'{out_dir}/ablation_summary_backup.csv', index=False, encoding='utf-8-sig')

    # 最终保存所有 7 个被试的综合汇总表到专属目录
    df_all = pd.DataFrame(all_results)
    final_csv_path = f'{out_dir}/7_subjects_ablation_summary.csv'
    df_all.to_csv(final_csv_path, index=False, encoding='utf-8-sig')

    print(f"\n🎉 【{current_ablation_mode}】模式的全部实验运行完毕！")
    print(f"📊 最终统计表已保存至: {final_csv_path}")


if __name__ == "__main__":
    main()