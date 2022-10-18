import os
import random

import pandas as pd
import torch
from torch.utils.data import Dataset
from shutil import copyfile


def label2Number(labels):
    # 标签门类
    labelCat = ['LAD', 'LCX', 'RCA', 'normal', 'other']
    dict1 = {'LAD': 0, 'LCX': 1, 'RCA': 2, 'normal': 3, 'other': 4}

    labelsCoeds = [dict1[str] for str in labels]
    return labelsCoeds


def write_filenames_to_txt(dir_path, txt_name):
    filenames = os.listdir(dir_path)
    with open(txt_name, 'w') as f:
        for filename in filenames:
            f.write(filename + "\n")


def sort_files_by_patient(source_dir, target_dir):
    source_files = os.listdir(source_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for source_file in source_files:
        label_name, patient_num, beat_num = source_file.split('.')[0].split('_')
        label_dir_path = os.path.join(target_dir, label_name)
        if not os.path.exists(label_dir_path):
            os.makedirs(label_dir_path)
        patient_dir_path = os.path.join(label_dir_path, patient_num)
        if not os.path.exists(patient_dir_path):
            os.makedirs(patient_dir_path)
        target_file_path = os.path.join(patient_dir_path, source_file)
        copyfile(os.path.join(source_dir, source_file), target_file_path)


def cp_rename_file(source_dir, target_dir):
    source_files = os.listdir(source_dir)
    count = 0
    for source_file in source_files:
        filename, file_type = source_file.split(".")
        target_file = filename + "_filter." + file_type
        target_path = os.path.join(target_dir, target_file)
        source_path = os.path.join(source_dir, source_file)
        copyfile(source_path, target_path)
        count += 1
    print(count)


def random_pick(source_dir, target_dir, sample_rate):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    label_dirs = os.listdir(source_dir)
    for label_dir in label_dirs:
        label_dir_path = os.path.join(source_dir, label_dir)
        patient_dirs = os.listdir(label_dir_path)
        num_of_sample = int(len(patient_dirs) * sample_rate)
        sampled_patients = random.sample(patient_dirs, num_of_sample)
        for sampled_patient in sampled_patients:
            sampled_patient_path = os.path.join(label_dir_path, sampled_patient)
            source_files = os.listdir(sampled_patient_path)
            for source_file in source_files:
                target_path = os.path.join(target_dir, source_file)
                source_path = os.path.join(sampled_patient_path, source_file)
                copyfile(source_path, target_path)


def pick_bad_files(source_dir):
    filenames = os.listdir(source_dir)
    for filename in filenames:
        try:
            sample_data = pd.read_csv(os.path.join(source_dir, filename), header=None)
        except Exception:
            print(filename)


def getDataFromDir(dirPath):
    # 请先输入文件目录
    file_list = os.listdir(dirPath)
    sample_name = file_list[0]
    sample_names = file_list
    labels = [str.split('_')[0] for str in sample_names]

    labels = label2Number(labels)  # 数值标签化

    # header=None：第一列不要成为列的名字
    samples_data = []
    for sample_name in sample_names:
        sample_data = pd.read_csv(os.path.join(dirPath, sample_name), header=None)
        sample_data = sample_data.values.T

        #使用reshape后，shape->[12,1200]->[120,120]
        sample_data = sample_data.reshape(120, 120)
        samples_data.append(sample_data)

    # samples_data和labels构成了训练集 接下来就把他们都转换成tensor
    samples_data = torch.tensor(samples_data).detach()

    #增加一个维度[w,h]->[channel,w,h]
    samples_data = torch.unsqueeze(samples_data, dim=1)

    labels = torch.tensor(labels).detach()
    print('文件路径:\n', dirPath)
    print('当前样本size:', samples_data.size(), '\n')
    print('当前标签size:', labels.size(), '\n')
    return samples_data, labels
# 修改了


class EcgDataset(Dataset):
    def __init__(self, data_path, network_type):
        self.data_path = data_path
        self.filenames = os.listdir(data_path)
        self.network_type = network_type

    def __getitem__(self, item):
        filename = self.filenames[item]
        sample_data = pd.read_csv(os.path.join(self.data_path, filename), header=None)
        sample_data = sample_data.values.T

        label_str = filename.split('_')[0]
        label_dict = {'LAD': 0, 'LCX': 1, 'RCA': 2, 'normal': 3, 'other': 4}
        label_num = label_dict[label_str]
        label_t = torch.tensor(label_num).detach()

        if self.network_type == "densenet":
            # 使用reshape后，shape->[12,1200]->[120,120]
            sample_data = sample_data.reshape(120, 120)

        # samples_data和labels构成了训练集 接下来就把他们都转换成tensor
        sample_data = torch.tensor(sample_data).detach().float()
        if self.network_type == "ecg_net":
            pass
        else:
            # 增加一个维度[w,h]->[channel,w,h]
            sample_data = torch.unsqueeze(sample_data, dim=0)

        return sample_data, label_t

    def __len__(self):
        return len(self.filenames)


if __name__ == "__main__":
    train_data_path = 'data/origin/origin_filter_train'
    # patient_data_path = 'data/origin/patient_filter'
    # target_path = 'data/origin/half_patient_filter'
    # random_pick(patient_data_path, target_path, 0.5)
    # sort_files_by_patient(train_data_path, patient_data_path)
    # target_dir_path = 'data/origin/half_origin_baseline_train'
    # random_pick(train_data_path, target_dir_path, 0.5)
    # write_filenames_to_txt(train_data_path, 'm8kd.txt')
    # pick_bad_files(train_data_path)
    # train_data = EcgDataset(train_data_path, "classic")
    # print(len(train_data))
    # d, l = train_data[0]
    # print(l)
    # target_dir = 'data/mix8000pf'
    target_dir = 'data/mix15000pf'
    cp_rename_file(train_data_path, target_dir)
