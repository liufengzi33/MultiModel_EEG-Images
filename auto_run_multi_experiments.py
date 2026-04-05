import os
import gc
import torch
import pandas as pd
import itertools
from config.config_multi_model import Config
from train_multimodal_three_phase import MultiStageTrainer


def main():
    # =====================================================================
    # 🎯 自动化消融模式列表
    # 脚本将依次遍历这些模式，全部跑完！
    # =====================================================================
    # ablation_modes = ['none', 'baseline_concat', 'no_cmd', 'no_ortho', ]
    # ablation_modes = ['baseline_concat', 'no_cmd', 'no_ortho', ]
    ablation_modes = ['baseline_concat']
    subjects = ["01gh", "02szy", "03ysq", "04whx", "05ly", "06wrl", "07lxy"]

    # 模型测试范围
    eeg_models = ['EEGNetv1', 'EEGNetv4', 'ShallowFBCSPNet']
    image_models = ['AlexNet', 'PlacesNet', 'VGG']
    image_types = ['sscnn', 'rsscnn']

    # 生成组合：EEG x Image x Type
    combinations = list(itertools.product(eeg_models, image_models, image_types))

    total_experiments = len(ablation_modes) * len(subjects) * len(combinations)
    print(f"🚀 准备开始【全自动消融实验】，共计 {len(ablation_modes)} 种模式！")
    print(f"📈 预计总计将运行 {total_experiments} 组实验组合。\n")

    # 用于收集【所有模式+所有被试】的终极汇总数据
    master_all_results = []

    # 🌟 最外层循环：遍历消融模式
    for current_ablation_mode in ablation_modes:
        print(f"\n{'#' * 85}")
        print(
            f"🔥 当前进入消融模式: 【{current_ablation_mode}】 ({ablation_modes.index(current_ablation_mode) + 1}/{len(ablation_modes)})")
        print(f"{'#' * 85}")

        # 动态创建该模式的专属输出目录
        out_dir = f'outputs/multi/{current_ablation_mode}'
        os.makedirs(out_dir, exist_ok=True)

        # 用于收集【当前模式下】所有被试的数据
        mode_all_results = []

        # 🌟 中层循环：遍历被试
        for subject_id in subjects:
            print(f"\n{'=' * 80}")
            print(f"👤 开始处理被试: {subject_id} (当前模式: {current_ablation_mode})")
            print(f"{'=' * 80}")

            subject_results = []

            # 🌟 内层循环：遍历模型组合
            for i, (eeg, img, img_type) in enumerate(combinations):
                print(f"\n{'-' * 70}")
                print(
                    f"🧪 实验 {i + 1}/{len(combinations)}: Sub={subject_id} | EEG={eeg} | Img={img} | Type={img_type}")
                print(f"{'-' * 70}")

                config = Config()
                config.subject_id = subject_id
                config.base_eeg_model = eeg
                config.base_image_model = img
                config.image_model_type = img_type
                config.ablation_mode = current_ablation_mode  # 动态注入当前消融模式

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
                        'Ablation_Mode': current_ablation_mode,  # 记录消融条件
                        'Subject_ID': subject_id,
                        'EEG_Model': eeg,
                        'Image_Model': img,
                        'Image_Type': img_type,
                        'Best_Epoch': best_epoch_idx + 1,
                        'Val_Acc(%)': round(best_acc, 2),
                        'Val_F1': round(best_f1, 4),
                        'Val_AUC': round(best_auc, 4),
                        'Timestamp': timestamp
                    }

                    # 同步追加到各个层级的列表中
                    subject_results.append(result_dict)
                    mode_all_results.append(result_dict)
                    master_all_results.append(result_dict)

                except Exception as e:
                    print(f"❌ 运行组合发生错误: {e}")
                    error_dict = {
                        'Ablation_Mode': current_ablation_mode, 'Subject_ID': subject_id,
                        'EEG_Model': eeg, 'Image_Model': img, 'Image_Type': img_type,
                        'Best_Epoch': 'Error', 'Val_Acc(%)': 'Error', 'Val_F1': 'Error', 'Val_AUC': 'Error',
                        'Timestamp': 'Error'
                    }
                    subject_results.append(error_dict)
                    mode_all_results.append(error_dict)
                    master_all_results.append(error_dict)

                # 强制清理显存
                del trainer
                if 'model' in locals():
                    del model
                gc.collect()
                torch.cuda.empty_cache()

            # 实时保存: 当前模式下，当前被试的汇总表
            df_subj = pd.DataFrame(subject_results)
            csv_subj_path = f'{out_dir}/{subject_id}_ablation_summary.csv'
            df_subj.to_csv(csv_subj_path, index=False, encoding='utf-8-sig')

            # 实时备份: 当前模式下，迄今为止所有被试的数据
            pd.DataFrame(mode_all_results).to_csv(f'{out_dir}/ablation_summary_backup.csv', index=False,
                                                  encoding='utf-8-sig')

            # 实时终极备份: 防止中途彻底断电，随时保存一份最全的数据
            pd.DataFrame(master_all_results).to_csv('outputs/multi/MASTER_ablation_summary_backup.csv', index=False,
                                                    encoding='utf-8-sig')

        # 保存: 当前消融模式跑完 7 个被试后的最终表
        df_mode_all = pd.DataFrame(mode_all_results)
        final_mode_csv_path = f'{out_dir}/7_subjects_{current_ablation_mode}_summary.csv'
        df_mode_all.to_csv(final_mode_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 【{current_ablation_mode}】模式的实验已全部完成并保存至: {final_mode_csv_path}")

    # 🌟 所有模式循环结束，生成终极汇总大表
    df_master = pd.DataFrame(master_all_results)
    master_csv_path = 'outputs/multi/MASTER_ALL_ABLATIONS_SUMMARY.csv'
    df_master.to_csv(master_csv_path, index=False, encoding='utf-8-sig')

    print(f"\n🎉🎉🎉 全部 {total_experiments} 组消融实验圆满结束！")
    print(f"🏆 终极大表已生成，包含了所有消融模式和被试的数据: {master_csv_path}")


if __name__ == "__main__":
    main()