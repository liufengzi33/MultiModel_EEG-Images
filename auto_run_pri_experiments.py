import os
import gc
import torch
import pandas as pd
import itertools

# 引入特权学习的配置和训练器
from config.config_privileged_model import PrivilegedConfig
from train_previlieged_modal import PrivilegedTrainer


def main():
    # 1. 定义需要遍历的参数字典
    eeg_models = ['EEGNetv1', 'EEGNetv4', 'ShallowFBCSPNet']
    image_models = ['AlexNet', 'PlacesNet', 'VGG']
    image_types = ['sscnn', 'rsscnn']

    # 生成所有 3 x 3 x 2 = 18 种组合
    combinations = list(itertools.product(eeg_models, image_models, image_types))

    # 获取被试ID用于命名
    temp_config = PrivilegedConfig()
    subject_id = temp_config.subject_id

    print(f"🚀 准备开始特权学习自动化运行，被试: {subject_id}，总计 {len(combinations)} 种实验组合。")

    results_list = []

    # 2. 依次运行组合
    for i, (eeg, img, img_type) in enumerate(combinations):
        print(f"\n{'=' * 70}")
        print(
            f"🧪 特权实验 {i + 1}/{len(combinations)}: Subject={subject_id} | EEG={eeg} | Image={img} | Type={img_type}")
        print(f"{'=' * 70}")

        # 动态修改配置
        config = PrivilegedConfig()
        config.eeg_model_name = eeg
        config.image_model_name = img
        config.image_model_type = img_type
        # 默认使用配置里的 student_modality = 'eeg'

        # 初始化 Trainer
        trainer = PrivilegedTrainer(config)

        try:
            # 启动训练
            trainer.train()

            # 3. 提取评估指标
            # 特权学习的记录都在 trainer.history 中
            val_losses = trainer.history['val_losses']
            # 以验证集总 Loss 最低的 epoch 作为最佳 Epoch
            best_epoch_idx = val_losses.index(min(val_losses))

            # 重点提取学生网络的指标和 KRR
            best_student_acc = trainer.history['val_student_acc'][best_epoch_idx]
            best_student_f1 = trainer.history['val_student_f1'][best_epoch_idx]
            best_student_auc = trainer.history['val_student_auc'][best_epoch_idx]
            best_teacher_acc = trainer.history['val_teacher_acc'][best_epoch_idx]
            best_krr = trainer.history['val_krr'][best_epoch_idx]

            # trainer.output_dir 类似于 outputs/outputs_privileged/.../01gh/20231024_123456
            # 我们取最后一段作为时间戳
            timestamp = os.path.basename(trainer.output_dir)

            results_list.append({
                'Subject_ID': subject_id,
                'EEG_Model': eeg,
                'Image_Model': img,
                'Image_Type': img_type,
                'Best_Epoch': best_epoch_idx + 1,
                'Teacher_Acc(%)': round(best_teacher_acc, 2),
                'Student_Acc(%)': round(best_student_acc, 2),
                'Student_F1': round(best_student_f1, 4),
                'Student_AUC': round(best_student_auc, 4),
                'KRR(%)': round(best_krr, 2),
                'Timestamp': timestamp,
                'Save_Dir': trainer.output_dir
            })

        except Exception as e:
            print(f"❌ 运行组合 {eeg} + {img} + {img_type} 时发生错误: {e}")
            results_list.append({
                'Subject_ID': subject_id, 'EEG_Model': eeg, 'Image_Model': img, 'Image_Type': img_type,
                'Best_Epoch': 'Error', 'Teacher_Acc(%)': 'Error', 'Student_Acc(%)': 'Error',
                'Student_F1': 'Error', 'Student_AUC': 'Error', 'KRR(%)': 'Error',
                'Timestamp': 'Error', 'Save_Dir': 'Error'
            })

        # 4. 强制清理内存与显存
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    # 5. 保存并输出最终统计表格
    df = pd.DataFrame(results_list)
    os.makedirs('outputs/pri', exist_ok=True)
    # 按被试ID命名CSV
    csv_path = f'outputs/{subject_id}_18_combinations_privileged_summary.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"\n🎉 被试 {subject_id} 的特权学习全部 {len(combinations)} 组实验运行完毕！")
    print(f"📊 汇总统计表已保存至: {csv_path}")
    print("\n📝 最终结果预览:")
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()