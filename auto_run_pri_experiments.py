import os
import gc
import torch
import pandas as pd
import itertools

# 引入特权学习的配置和训练器
from config.config_privileged_model import PrivilegedConfig
from train_previlieged_modal import PrivilegedTrainer


def main():
    # 1. 定义7个被试的编号
    subjects = ["01gh", "02szy", "03ysq", "04whx", "05ly", "06wrl", "07lxy"]

    # 2. 定义需要遍历的参数字典
    eeg_models = ['EEGNetv1', 'EEGNetv4', 'ShallowFBCSPNet']
    image_models = ['AlexNet', 'PlacesNet', 'VGG']
    image_types = ['sscnn', 'rsscnn']

    # 生成所有 3 x 3 x 2 = 18 种组合
    combinations = list(itertools.product(eeg_models, image_models, image_types))

    print(f"🚀 准备开始特权学习多被试自动化运行，总计 {len(subjects) * len(combinations)} 种实验组合。")

    all_results = []
    os.makedirs('outputs/pri', exist_ok=True)

    # 3. 外层循环：遍历被试
    for subject_id in subjects:
        print(f"\n{'=' * 80}")
        print(f"👤 开始处理被试: {subject_id}")
        print(f"{'=' * 80}")

        subject_results = []

        # 4. 内层循环：依次运行组合
        for i, (eeg, img, img_type) in enumerate(combinations):
            print(f"\n{'=' * 70}")
            print(
                f"🧪 特权实验 {i + 1}/{len(combinations)}: Subject={subject_id} | EEG={eeg} | Image={img} | Type={img_type}")
            print(f"{'=' * 70}")

            # 动态修改配置
            config = PrivilegedConfig()
            config.subject_id = subject_id  # 动态注入被试ID
            config.eeg_model_name = eeg
            config.image_model_name = img
            config.image_model_type = img_type

            # 初始化 Trainer
            trainer = PrivilegedTrainer(config)

            try:
                # 启动训练
                trainer.train()

                # 提取评估指标
                val_losses = trainer.history['val_losses']
                best_epoch_idx = val_losses.index(min(val_losses))

                # 重点提取学生与教师网络指标
                best_student_acc = trainer.history['val_student_acc'][best_epoch_idx]
                best_student_f1 = trainer.history['val_student_f1'][best_epoch_idx]
                best_student_auc = trainer.history['val_student_auc'][best_epoch_idx]

                best_teacher_acc = trainer.history['val_teacher_acc'][best_epoch_idx]
                best_teacher_f1 = trainer.history['val_teacher_f1'][best_epoch_idx]
                best_teacher_auc = trainer.history['val_teacher_auc'][best_epoch_idx]

                best_krr = trainer.history['val_krr'][best_epoch_idx]

                timestamp = os.path.basename(trainer.output_dir)

                result_dict = {
                    'Subject_ID': subject_id,
                    'EEG_Model': eeg,
                    'Image_Model': img,
                    'Image_Type': img_type,
                    'Best_Epoch': best_epoch_idx + 1,
                    'Teacher_Acc(%)': round(best_teacher_acc, 2),
                    'Teacher_F1': round(best_teacher_f1, 4),
                    'Teacher_AUC': round(best_teacher_auc, 4),
                    'Student_Acc(%)': round(best_student_acc, 2),
                    'Student_F1': round(best_student_f1, 4),
                    'Student_AUC': round(best_student_auc, 4),
                    'KRR(%)': round(best_krr, 2),
                    'Timestamp': timestamp,
                    'Save_Dir': trainer.output_dir
                }

                subject_results.append(result_dict)
                all_results.append(result_dict)

            except Exception as e:
                print(f"❌ 运行组合 {subject_id} + {eeg} + {img} + {img_type} 时发生错误: {e}")
                error_dict = {
                    'Subject_ID': subject_id, 'EEG_Model': eeg, 'Image_Model': img, 'Image_Type': img_type,
                    'Best_Epoch': 'Error',
                    'Teacher_Acc(%)': 'Error', 'Teacher_F1': 'Error', 'Teacher_AUC': 'Error',
                    'Student_Acc(%)': 'Error', 'Student_F1': 'Error', 'Student_AUC': 'Error',
                    'KRR(%)': 'Error', 'Timestamp': 'Error', 'Save_Dir': 'Error'
                }
                subject_results.append(error_dict)
                all_results.append(error_dict)

            # 强制清理内存与显存
            del trainer
            gc.collect()
            torch.cuda.empty_cache()

        # 5. 当前被试跑完，保存专属表格
        df_subj = pd.DataFrame(subject_results)
        csv_subj_path = f'outputs/pri/{subject_id}_18_combinations_privileged_summary.csv'
        df_subj.to_csv(csv_subj_path, index=False, encoding='utf-8-sig')
        print(f"💾 被试 {subject_id} 的专属统计表已保存至: {csv_subj_path}")

        # 实时全局备份
        pd.DataFrame(all_results).to_csv('outputs/pri/pri_summary_backup.csv', index=False, encoding='utf-8-sig')

    # 6. 保存并输出最终总表格
    df_all = pd.DataFrame(all_results)
    final_csv_path = 'outputs/pri/7_subjects_pri_summary.csv'
    df_all.to_csv(final_csv_path, index=False, encoding='utf-8-sig')

    print(f"\n🎉 全部 7 个被试的特权学习实验运行完毕！")
    print(f"📊 汇总统计表已保存至: {final_csv_path}")


if __name__ == "__main__":
    main()