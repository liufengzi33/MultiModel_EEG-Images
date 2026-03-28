import os
import gc
import torch
import pandas as pd
import itertools
from config.config_multi_model import Config
from train_multimodal_three_phase import MultiStageTrainer


def main():
    # 1. 定义7个被试的编号
    subjects = ["01gh", "02szy", "03ysq", "04whx", "05ly", "06wrl", "07lxy"]

    # 2. 定义需要遍历的参数字典
    eeg_models = ['EEGNetv1', 'EEGNetv4', 'ShallowFBCSPNet']
    image_models = ['AlexNet', 'PlacesNet', 'VGG']
    image_types = ['sscnn', 'rsscnn']

    # 生成所有 3 x 3 x 2 = 18 种组合
    combinations = list(itertools.product(eeg_models, image_models, image_types))

    print(f"🚀 准备开始多模态多被试自动化运行，总计 {len(subjects) * len(combinations)} 组实验组合。")

    all_results = []
    os.makedirs('outputs/multi', exist_ok=True)

    # 3. 外层循环：遍历被试
    for subject_id in subjects:
        print(f"\n{'=' * 80}")
        print(f"👤 开始处理被试: {subject_id}")
        print(f"{'=' * 80}")

        subject_results = []

        # 4. 内层循环：遍历模型组合
        for i, (eeg, img, img_type) in enumerate(combinations):
            print(f"\n{'=' * 70}")
            print(
                f"🧪 实验 {i + 1}/{len(combinations)}: Subject={subject_id} | EEG={eeg} | Image={img} | Type={img_type}")
            print(f"{'=' * 70}")

            # 动态修改配置
            config = Config()
            config.subject_id = subject_id  # 动态注入被试ID
            config.base_eeg_model = eeg
            config.base_image_model = img
            config.image_model_type = img_type

            # 初始化 Trainer
            trainer = MultiStageTrainer(config)
            trainer.setup_data()
            trainer.setup_model()

            try:
                # 启动训练
                model, history = trainer.train()

                # 提取评估指标
                val_losses = history['all_val_loss']
                best_epoch_idx = val_losses.index(min(val_losses))

                best_acc = history['all_val_acc'][best_epoch_idx]
                best_f1 = history['all_val_f1'][best_epoch_idx]
                best_auc = history['all_val_auc'][best_epoch_idx]

                timestamp = trainer.save_dir.split('multimodal_')[-1]

                result_dict = {
                    'Subject_ID': subject_id,
                    'EEG_Model': eeg,
                    'Image_Model': img,
                    'Image_Type': img_type,
                    'Best_Epoch': best_epoch_idx + 1,
                    'Val_Acc(%)': round(best_acc, 2),
                    'Val_F1': round(best_f1, 4),
                    'Val_AUC': round(best_auc, 4),
                    'Timestamp': timestamp,
                    'Save_Dir': trainer.save_dir
                }

                subject_results.append(result_dict)
                all_results.append(result_dict)

            except Exception as e:
                print(f"❌ 运行组合 {subject_id} + {eeg} + {img} + {img_type} 时发生错误: {e}")
                error_dict = {
                    'Subject_ID': subject_id, 'EEG_Model': eeg, 'Image_Model': img, 'Image_Type': img_type,
                    'Best_Epoch': 'Error', 'Val_Acc(%)': 'Error',
                    'Val_F1': 'Error', 'Val_AUC': 'Error',
                    'Timestamp': 'Error', 'Save_Dir': 'Error'
                }
                subject_results.append(error_dict)
                all_results.append(error_dict)

            # 强制清理内存与显存
            del trainer
            if 'model' in locals():
                del model
            gc.collect()
            torch.cuda.empty_cache()

        # 5. 当前被试跑完，保存专属表格
        df_subj = pd.DataFrame(subject_results)
        csv_subj_path = f'outputs/multi/{subject_id}_18_combinations_summary.csv'
        df_subj.to_csv(csv_subj_path, index=False, encoding='utf-8-sig')
        print(f"💾 被试 {subject_id} 的专属统计表已保存至: {csv_subj_path}")

        # 实时全局备份
        pd.DataFrame(all_results).to_csv('outputs/multi/multi_summary_backup.csv', index=False, encoding='utf-8-sig')

    # 6. 保存并输出最终总表格
    df_all = pd.DataFrame(all_results)
    final_csv_path = 'outputs/multi/7_subjects_multi_summary.csv'
    df_all.to_csv(final_csv_path, index=False, encoding='utf-8-sig')

    print(f"\n🎉 全部 7 个被试的多模态实验运行完毕！")
    print(f"📊 汇总统计表已保存至: {final_csv_path}")


if __name__ == "__main__":
    main()