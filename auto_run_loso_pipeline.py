import os
import gc
import torch
import pandas as pd
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

# 导入配置和模型
from config.config_eeg_model import Config as EEGConfig
from config.config_multi_model import Config as MultiConfig
from MyPP2Dataset import MyPP2Dataset
from models.eeg_models import SSBCINet
from train_eegs import EEGTrainer
from train_multimodal_three_phase import MultiStageTrainer


def get_balanced_loader(dataset, batch_size):
    """【工具函数】为合并后的跨被试数据集计算类别权重，并返回带有 Sampler 的 DataLoader"""
    print("  -> 正在统计合并数据集的标签分布，计算 Sampler 权重...")
    train_labels = [dataset[i][4] for i in range(len(dataset))]
    train_counts = torch.bincount(torch.tensor(train_labels)).tolist()
    print(f"  -> 全局训练集类别分布: {train_counts}")

    class_weights = 1.0 / torch.tensor(train_counts, dtype=torch.float)
    sample_weights = [class_weights[int(label)] for label in train_labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False
    )
    return loader


def run_phase_a_eeg_pretrain(all_subjects, eeg_models):
    """
    阶段一：运行所有 LOSO 折数的 EEG 单模态预训练，并保存到本地
    """
    print(f"\n{'=' * 70}")
    print(f"🚀 [Phase A] 启动跨被试 EEG 批量预训练")
    print(f"待跑模型: {eeg_models}")
    print(f"{'=' * 70}")

    results = []
    os.makedirs('outputs/loso', exist_ok=True)
    csv_path = 'outputs/loso/master_phase_a_eeg_pretrain.csv'

    # 🌟 优化点：外层循环是被试，内层循环是模型。这样数据集只需构建一次！
    for test_subject in all_subjects:
        train_subjects = [s for s in all_subjects if s != test_subject]
        loso_fold_id = f"LOSO_test_{test_subject}"  # 虚拟 ID，用于目录隔离

        print(f"\n⚡ 当前 Fold 测试集: 【{test_subject}】 | 训练集: 剩余 6 人")
        print(f"📦 正在构建合并数据集 (此折数只需加载一次数据)...")

        # 1. 构造数据集 (只需一次)
        train_datasets = [MyPP2Dataset(is_flipped=False, subject_id=s) for s in train_subjects]
        combined_train_ds = ConcatDataset(train_datasets)
        test_ds = MyPP2Dataset(is_flipped=False, subject_id=test_subject)

        for eeg_model_name in eeg_models:
            print(f"\n  🧠 正在预训练 EEG 模型: {eeg_model_name}")

            # 2. 配置与 DataLoader
            cfg = EEGConfig()
            cfg.subject_id = loso_fold_id
            cfg.base_model_name = eeg_model_name

            train_loader = get_balanced_loader(combined_train_ds, cfg.batch_size)
            test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

            # 3. 初始化模型与训练器
            model = SSBCINet(base_model_name=cfg.base_model_name, in_chans=64, n_classes=2, input_window_samples=2000)

            trainer = EEGTrainer(
                model=model, train_loader=train_loader, val_loader=test_loader,
                device=cfg.device, lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
                patience=cfg.patience, factor=cfg.factor, max_plateaus=cfg.max_lr_plateaus,
                base_model_name=cfg.base_model_name, subject_id=cfg.subject_id,
                pos_weight=None  # 使用了 Sampler，必须为 None
            )

            # 4. 执行训练
            best_f1, best_acc, best_epoch = trainer.train(epochs=cfg.num_epochs)

            results.append({
                'Test_Subject': test_subject,
                'Fold_ID': loso_fold_id,
                'EEG_Model': eeg_model_name,
                'Best_Acc(%)': round(best_acc, 2),
                'Best_F1': round(best_f1, 4),
                'Best_Epoch': best_epoch
            })

            # 实时保存结果
            pd.DataFrame(results).to_csv(csv_path, index=False, encoding='utf-8-sig')

            # 5. 暴力清空显存，准备迎接下一个网络
            del model, trainer, train_loader, test_loader
            gc.collect()
            torch.cuda.empty_cache()

        # 当前 Fold 的所有模型跑完，清理数据集内存
        del combined_train_ds, test_ds
        gc.collect()

    print(f"\n✅ Phase A 全部完成！批量预训练结果已保存至: {csv_path}")


def run_phase_b_multimodal_finetune(all_subjects, eeg_models, image_model_name, img_type):
    """
    阶段二：加载本地的 EEG 预训练权重，运行双模态网络
    """
    print(f"\n{'=' * 70}")
    print(f"🧬 [Phase B] 启动跨被试双模态批量融合训练")
    print(f"待跑 EEG Backbone: {eeg_models}")
    print(f"{'=' * 70}")

    results = []
    csv_path = 'outputs/loso/master_phase_b_multimodal.csv'

    # 同理，被试循环在外，模型循环在内
    for test_subject in all_subjects:
        train_subjects = [s for s in all_subjects if s != test_subject]
        loso_fold_id = f"LOSO_test_{test_subject}"

        print(f"\n🔍 当前 Fold 测试集: 【{test_subject}】 | 准备数据...")

        # 1. 构造数据集
        train_datasets = [MyPP2Dataset(is_flipped=False, subject_id=s) for s in train_subjects]
        combined_train_ds = ConcatDataset(train_datasets)
        test_ds = MyPP2Dataset(is_flipped=False, subject_id=test_subject)

        for eeg_model_name in eeg_models:
            print(f"\n  🔗 正在微调多模态网络 (EEG Backbone: {eeg_model_name})")

            # 2. 配置双模态参数
            cfg = MultiConfig()
            cfg.subject_id = loso_fold_id  # 这一步决定了 ModelLoader 会去哪个文件夹找模型
            cfg.base_eeg_model = eeg_model_name
            cfg.base_image_model = image_model_name
            cfg.image_model_type = img_type
            cfg.use_pretrained_eeg = True
            cfg.use_pretrained_image = True

            trainer = MultiStageTrainer(cfg)

            # 手动覆盖 DataLoader (此处不用 Sampler，按你原代码逻辑用 shuffle)
            trainer.train_loader = DataLoader(combined_train_ds, batch_size=cfg.batch_size, shuffle=True)
            trainer.test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

            # 3. 初始化模型 (内部的 ModelLoader 会根据 cfg.subject_id 和 eeg_model_name 精准找到 Phase A 权重)
            try:
                trainer.setup_model()

                # 4. 执行三阶段训练
                _, history = trainer.train()

                val_accs = history['all_val_acc']
                best_idx = val_accs.index(max(val_accs))

                results.append({
                    'Test_Subject': test_subject,
                    'EEG_Model': eeg_model_name,
                    'Image_Model': image_model_name,
                    'Type': img_type,
                    'Best_Val_Acc(%)': round(history['all_val_acc'][best_idx], 2),
                    'Best_Val_F1': round(history['all_val_f1'][best_idx], 4),
                    'Best_Val_AUC': round(history['all_val_auc'][best_idx], 4)
                })
                print(f"  🎉 当前组合完成: Acc={results[-1]['Best_Val_Acc(%)']}%")

            except Exception as e:
                print(f"  ❌ 运行失败: {e}")
                results.append({
                    'Test_Subject': test_subject, 'EEG_Model': eeg_model_name,
                    'Image_Model': image_model_name, 'Type': img_type,
                    'Best_Val_Acc(%)': 'Error', 'Best_Val_F1': 'Error', 'Best_Val_AUC': 'Error'
                })

            # 实时保存
            pd.DataFrame(results).to_csv(csv_path, index=False, encoding='utf-8-sig')

            # 5. 清空显存
            del trainer
            gc.collect()
            torch.cuda.empty_cache()

        # 清理当前 Fold 数据集
        del combined_train_ds, test_ds
        gc.collect()

    print(f"\n✅ Phase B 全部完成！双模态结果汇总已保存至: {csv_path}")


def main():
    all_subjects = ["01gh", "02szy", "03ysq", "04whx", "05ly", "06wrl", "07lxy"]

    # 🌟 将 EEG 模型改为列表
    eeg_models = ['EEGNetv1', 'EEGNetv4', 'ShallowFBCSPNet']

    image_model_name = 'VGG'
    img_type = 'sscnn'

    # ---------------------------------------------------------

    # 流程控制面板：你可以通过注释下方代码，单独跑阶段 A 或 B

    # ---------------------------------------------------------


    # 步骤 1：批量跑通所有折数、所有 EEG 架构的预训练，并落盘
    # run_phase_a_eeg_pretrain(all_subjects, eeg_models)

    # 步骤 2：读取本地所有 EEG 权重，批量跑所有折数的双模态训练
    run_phase_b_multimodal_finetune(all_subjects, eeg_models, image_model_name, img_type)


if __name__ == "__main__":
    main()