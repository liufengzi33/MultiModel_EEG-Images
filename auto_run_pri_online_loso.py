import os
import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils.data import ConcatDataset, DataLoader

# 导入你现有的配置、Trainer和数据集
from config.config_privileged_model import get_config
from train_previlieged_modal import PrivilegedTrainer
from MyPP2Dataset import MyPP2Dataset


class LOSOPrivilegedTrainer(PrivilegedTrainer):
    """
    继承原有的 PrivilegedTrainer，仅重写数据加载逻辑，实现跨被试 LOSO 切分
    """

    def __init__(self, config, all_subjects, test_subject):
        self.all_subjects = all_subjects
        self.test_subject = test_subject
        super().__init__(config)

    def create_dataloaders(self):
        print(f"\n📦 创建 LOSO 数据加载器... 验证集被试 (Test): {self.test_subject}")

        # 数据集严格划分，防图像特征泄露
        # 训练集: 0-239 (前80%) | 测试集: 240-299 (后20%)
        train_indices = list(range(0, 240))
        test_indices = list(range(240, 300))

        train_datasets = []
        for subj in self.all_subjects:
            if subj == self.test_subject:
                continue

            # 训练集：提取其余 6 个被试的 前 240 个样本
            ds = MyPP2Dataset(
                is_flipped=False,
                transform=self.config.transform,
                subject_id=subj,
                allowed_indices=train_indices
            )
            train_datasets.append(ds)

        # 拼接 6 个被试的训练数据 (6 * 240 = 1440 个样本)
        train_dataset = ConcatDataset(train_datasets)

        # 测试集：仅提取当前测试被试的 后 60 个样本
        test_dataset = MyPP2Dataset(
            is_flipped=False,
            transform=self.config.transform,
            subject_id=self.test_subject,
            allowed_indices=test_indices
        )

        print(f"   => 最终训练集样本数: {len(train_dataset)} | 最终测试集样本数: {len(test_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)

        return train_loader, val_loader


def main():
    # ================= 配置区域 =================
    all_subjects = ["01gh", "02szy", "03ysq", "04whx", "05ly", "06wrl", "07lxy"]
    MODE = "Cross-subject"

    # 统计数据 CSV 保存路径 (按要求保存到 outputs/pri/Cross_sub)
    stats_root = os.path.join("outputs", "pri", "Cross_sub")
    os.makedirs(stats_root, exist_ok=True)

    # 模型权重与图表保存根路径 (按要求在 outputs/privileged 下)
    weights_root = os.path.join("outputs", "privileged", "Cross_sub")
    # ============================================

    all_results = []
    print(f"🚀 启动 LOSO 跨被试在线蒸馏 | 模式: {MODE}")

    for test_subject in all_subjects:
        print(f"\n{'=' * 60}\n正在处理 LOSO 测试集被试: {test_subject}\n{'=' * 60}")

        # 获取配置并覆盖特定参数
        config = get_config('default')
        config.subject_id = test_subject
        config.train_mode = MODE

        # 生成与 offline 脚本类似的目录名
        model_subdir = f"{config.eeg_model_name}_{config.image_model_name}_{config.image_model_type}_student_{config.student_modality}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 权重保存路径：outputs/privileged/Cross_sub/[模型组合]/LOSO_test_01gh/[时间戳]
        final_out_dir = os.path.join(weights_root, model_subdir, f"LOSO_test_{test_subject}", timestamp)
        os.makedirs(final_out_dir, exist_ok=True)

        config.output_dir = final_out_dir

        # 实例化自定义的 LOSO Trainer 并启动训练
        trainer = LOSOPrivilegedTrainer(config, all_subjects, test_subject)
        trainer.train()

        # ================= 提取训练结果 =================
        # Trainer 运行结束后，直接从内存读取 history 提取最优指标
        history = trainer.history
        if history and len(history['val_student_acc']) > 0:
            # 以学生网络验证集准确率最高的那一轮为准
            best_idx = np.argmax(history['val_student_acc'])

            best_acc = history['val_student_acc'][best_idx]
            best_f1 = history['val_student_f1'][best_idx]
            best_auc = history['val_student_auc'][best_idx]
            teacher_acc = history['val_teacher_acc'][best_idx]
            krr = history['val_krr'][best_idx]

            res = {
                "Test Subject": test_subject,
                "Student Acc (%)": best_acc,
                "Student F1": best_f1,
                "Student AUC": best_auc,
                "Teacher Acc (%)": teacher_acc,
                "KRR (%)": krr
            }
            all_results.append(res)

            # 保存当前被试的独立 CSV 统计文件
            subject_csv_path = os.path.join(stats_root, f"{test_subject}_online_pri_distill.csv")
            pd.DataFrame([res]).to_csv(subject_csv_path, index=False)
            print(f"📊 被试 {test_subject} 统计已保存至: {subject_csv_path}")

    # ================= 汇总所有结果 =================
    if all_results:
        df = pd.DataFrame(all_results)

        # 自动计算均值行并追加到末尾，方便论文制表
        mean_res = df.mean(numeric_only=True).to_dict()
        mean_res["Test Subject"] = "Average"
        df = pd.concat([df, pd.DataFrame([mean_res])], ignore_index=True)

        # 保存总表
        summary_path = os.path.join(stats_root, f"TOTAL_summary_{MODE}.csv")
        df.to_csv(summary_path, index=False)
        print(f"\n✅ 所有 {len(all_subjects)} 个 LOSO 实验全部完成！总汇总表保存在: {summary_path}")


if __name__ == "__main__":
    main()