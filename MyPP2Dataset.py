import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
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
        is_flipped ([type]): [是否交换图像的左右顺序]
    """

    # TODO: 需要添加EEG信号的读取
    def __init__(self, csv_file="data/safe_qscores_high2low.xlsx",
                 transform=transform_cnn,
                 img_dir="data",
                 is_flipped=False):
        self.csv_file = csv_file
        self.transform = transform
        self.img_dir = img_dir
        self.is_flipped = is_flipped
        self.dataframe = pd.read_excel(self.csv_file)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        left_img = self.get_image_by_name(self.dataframe.iloc[idx]['left'])
        right_img = self.get_image_by_name(self.dataframe.iloc[idx]['right'])
        # TODO: 在这里添加EEG信号的读取 需要两个，翻转和不翻转的EEG信号
        label = self.dataframe.iloc[idx]['choice']

        if self.is_flipped:  # 翻转的话，左侧图像和右侧图像的标签也要翻转
            left_img, right_img = right_img, left_img
            label = 1 - label
        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
        return left_img, right_img, label

    def get_image_by_name(self, image_name):
        # 读取图像
        image_path = os.path.join(self.img_dir, image_name)
        image = io.imread(image_path)
        return image


if __name__ == "__main__":
    # 创建数据集实例
    dataset = MyPP2Dataset()

    # 获取数据集长度
    print(f"数据集长度: {len(dataset)}")

    # 随机选择一个索引
    idx = np.random.randint(0, len(dataset))

    # 获取样本数据
    left_img, right_img, label = dataset[idx]

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
