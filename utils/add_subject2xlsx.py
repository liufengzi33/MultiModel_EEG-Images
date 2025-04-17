import pandas as pd

def add_subject_to_xlsx(xlsx_filename, subject_id):
    """
    添加被试编号到xlsx文件中
    :param filename: xlsx文件名
    :param subject_id: 被试编号
    """
    # 读取已有的xlsx文件
    df = pd.read_excel(xlsx_filename)

    # 创建文件名列表
    segments_LR = [f"segment_{i:03d}.npy" for i in range(300)]
    segments_RL = [f"segment_{i:03d}.npy" for i in range(300, 600)]

    # 添加字段
    df[f"{subject_id}_LR"] = segments_LR
    df[f"{subject_id}_RL"] = segments_RL

    # 保存为新的文件或覆盖原文件
    df.to_excel(xlsx_filename, index=False)  # 可改名保存

if __name__ == '__main__':
    subjects_list = ['01gh', '02szy', '03ysq', '04whx', '05ly']
    xlsx_filename = "../data/safe_qscores_high2low.xlsx"  # 修改为你的xlsx文件名
    for subject in subjects_list:
        add_subject_to_xlsx(xlsx_filename, subject)