import os
import gc
import torch
import pandas as pd
import itertools
from config.config_multi_model import Config
from train_multimodal_three_phase import MultiStageTrainer


def main():
    # 1. 定义需要遍历的参数字典
    eeg_models = ['EEGNetv1', 'EEGNetv4', 'ShallowFBCSPNet']
    image_models = ['AlexNet', 'PlacesNet', 'VGG']
    image_types = ['sscnn', 'rsscnn']

    # 生成所有 3 x 3 x 2 = 18 种组合
    combinations = list(itertools.product(eeg_models, image_models, image_types))

    # 获取被试ID用于命名
    temp_config = Config()
    subject_id = temp_config.subject_id

    print(f"🚀 准备开始运行，被试: {subject_id}，总计 {len(combinations)} 种实验组合。")

    results_list = []

    # 2. 依次运行组合
    for i, (eeg, img, img_type) in enumerate(combinations):
        print(f"\n{'=' * 70}")
        print(f"🧪 实验 {i + 1}/{len(combinations)}: Subject={subject_id} | EEG={eeg} | Image={img} | Type={img_type}")
        print(f"{'=' * 70}")

        # 动态修改配置
        config = Config()
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

            # 3. 提取评估指标
            val_losses = history['all_val_loss']
            best_epoch_idx = val_losses.index(min(val_losses))

            best_acc = history['all_val_acc'][best_epoch_idx]
            best_f1 = history['all_val_f1'][best_epoch_idx]
            best_auc = history['all_val_auc'][best_epoch_idx]

            timestamp = trainer.save_dir.split('multimodal_')[-1]

            results_list.append({
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
            })

        except Exception as e:
            print(f"❌ 运行组合 {eeg} + {img} + {img_type} 时发生错误: {e}")
            results_list.append({
                'Subject_ID': subject_id, 'EEG_Model': eeg, 'Image_Model': img, 'Image_Type': img_type,
                'Best_Epoch': 'Error', 'Val_Acc(%)': 'Error',
                'Val_F1': 'Error', 'Val_AUC': 'Error',
                'Timestamp': 'Error', 'Save_Dir': 'Error'
            })

        # 4. 强制清理内存与显存
        del trainer
        if 'model' in locals():
            del model
        gc.collect()
        torch.cuda.empty_cache()

    # 5. 保存并输出最终统计表格
    df = pd.DataFrame(results_list)
    os.makedirs('outputs/multi', exist_ok=True)
    # 按被试ID命名CSV
    csv_path = f'outputs/{subject_id}_18_combinations_summary.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"\n🎉 被试 {subject_id} 的全部 {len(combinations)} 组实验运行完毕！")
    print(f"📊 汇总统计表已保存至: {csv_path}")


if __name__ == "__main__":
    main()