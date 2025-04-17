import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from pandas.core.accessor import register_dataframe_accessor
from torch.utils.data import Dataset
from skimage import io
from torchvision import transforms
import pandas as pd
from torchvision.transforms.functional import crop

import os
from os import listdir
from os.path import isfile, join


def crop_google_logo(img):
    return crop(img, 0, 0, img.size[1] - 25, img.size[0])  # D裁剪google logo底部25个像素


# 定义一种transforms
transform_cnn = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Lambda(crop_google_logo),
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
])


class MyPP2Dataset(Dataset):
    """
    实验数据构造pytorch数据集，读取data/safe_qscores_high2low.xlsx文件 共计300组实验对
    本实验仅包含study_id对应safe的实验
    label的对应关系：0->左图，1->右图，即标签为0表示左侧图更安全
    Args:
        csv_file ([csv xlsx etc.]): [实验数据文件]
        transform ([type]): [图像转换]
        image_dir ([type]): [图像存储路径]
        eeg_dir ([type]): [EEG信号存储路径]
        is_flipped ([type]): [是否交换图像的左右顺序]
        subject_id ([type]): [实验对象id]
    """

    def __init__(self, csv_file="data/safe_qscores_high2low.xlsx",
                 transform=transform_cnn,
                 img_dir="data",
                 eeg_dir="data/EEG/seg_eeg_data",
                 is_flipped=False,
                 subject_id="01gh"):
        self.csv_file = csv_file
        self.transform = transform
        self.img_dir = img_dir
        self.eeg_dir = eeg_dir
        self.is_flipped = is_flipped
        self.subject_id = subject_id

        self.dataframe = pd.read_excel(self.csv_file)
        self.eeg_col = f"{subject_id}_RL" if is_flipped else f"{subject_id}_LR"

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        left_img = self.get_image_by_name(row['left'])
        right_img = self.get_image_by_name(row['right'])
        label = row['choice']

        if self.is_flipped:  # 翻转的话，左侧图像和右侧图像的标签也要翻转
            left_img, right_img = right_img, left_img
            label = 1 - label
        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)

        # 读取eeg信号
        eeg_filename = row[self.eeg_col]  # segment_000.npy
        eeg_path = os.path.join(self.eeg_dir, self.subject_id, eeg_filename)
        eeg_data = np.load(eeg_path)
        # TODO 后续eeg_data可能也需要分成左图和右图对应的eeg 看看要不要转成float32的
        return left_img, right_img, eeg_data, label

    def get_image_by_name(self, image_name):
        # 读取图像
        image_path = os.path.join(self.img_dir, image_name)
        image = io.imread(image_path)
        return image


if __name__ == "__main__":
    # 创建数据集实例
    dataset = MyPP2Dataset(is_flipped=False)

    # 获取数据集长度
    print(f"数据集长度: {len(dataset)}")

    # 随机选择一个索引
    # idx = np.random.randint(0, len(dataset))
    idx = 0  # 选择第一个样本
    # 获取样本数据
    left_img, right_img, eeg_data, label = dataset[idx]

    # === 断言 EEG 数据是否与 segment_000.npy 文件一致 ===
    expected_eeg_path = os.path.join("data/EEG/seg_eeg_data/01gh", "segment_000.npy")
    expected_eeg = np.load(expected_eeg_path)

    assert np.allclose(eeg_data, expected_eeg), "EEG 数据与 segment_000.npy 不一致！"
    print("✅ EEG 数据正确无误，与 segment_000.npy 一致。")

    # 显示图片
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(left_img.permute(1, 2, 0))  # 还原通道格式 (H, W, C)
    axes[0].set_title("Left Image")
    axes[0].axis("off")

    axes[1].imshow(right_img.permute(1, 2, 0))
    axes[1].set_title("Right Image")
    axes[1].axis("off")

    # 设置标题，显示标签信息
    plt.suptitle(f"Label: {'Left safer' if label == 0 else 'Right safer'}")
    plt.show()
