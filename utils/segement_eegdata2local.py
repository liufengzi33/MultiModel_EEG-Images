import numpy as np
import os
from tqdm import tqdm
def segment_eeg_data_to_local(eeg_data, save_dir):
    """
    将EEG数据分段并保存到本地
    :param eeg_data: EEG数据，形状为 (600, 64, 6000)
    :param save_dir: 保存目录
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in tqdm(range(eeg_data.shape[0]), desc="Saving EEG segments"):
        # 获取每个实验的EEG数据
        segment = eeg_data[i]
        # 保存为.npy文件
        np.save(os.path.join(save_dir, f"segment_{i:03d}.npy"), segment)

if __name__ == '__main__':
    subjects_list = ['01gh', '02szy', '03ysq', '04whx']
    for subject in  subjects_list:
        # 读取EEG数据
        eeg_data_path = f"../data/EEG/raw_eeg_npy/{subject}.npy"
        save_dir = f"../data/EEG/seg_eeg_data/{subject}"
        # 加载EEG数据
        eeg_data = np.load(eeg_data_path)
        # 分段并保存
        segment_eeg_data_to_local(eeg_data, save_dir)