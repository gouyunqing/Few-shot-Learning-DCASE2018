import torch
import numpy as np
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from Config import Config
import pickle
import pandas as pd


# 下面两个函数 ，实现，将一个二维矩阵 补零到 指定长度。 （补一列一列的零）. 如果超过 指定的seglen，则切掉多余的。
def pad(x, seglen, mode='wrap'):
    pad_len = seglen - x.shape[1]
    y = np.pad(x, ((0,0), (0,pad_len)), mode=mode)
    return y


def segment(x, seglen=1024):
    '''
    :param x: npy形式的mel [80,L]
    :param seglen: padding长度
    :return: padding mel
    '''
    ## 该函数将melspec [80,len] ，padding到固定长度 seglen
    if x.shape[1] < seglen:
        y = pad(x, seglen)
    elif x.shape[1] == seglen:
        y = x
    else:
        r = np.random.randint(x.shape[1] - seglen) ## r : [0-  (L-128 )]
        y = x[:, r:r+seglen]
    return y


def read_data_path_txt(txt_path):
    data_path_list = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data_path_list.append(Path(line.strip('\n')))
    return data_path_list


class MelDataset(Dataset):
    def __init__(self, txt_path_list: list, seq_len, shot=None, query=None):
        '''
        Args:
            data_path_txt: 存放数据路径的txt，如train.txt包含了support set和query set中会抽取到的所有数据路径
            seq_len: 需要pad到seq
            shot: support set中的数据量(在本代码中默认shot=batch_size)
            query: query set中的数据量(在本代码中默认shot=batch_size)
        '''
        self.data_list = txt_path_list
        if len(txt_path_list) == 4:
            self.mode = 'train'
            self.support_positive_data_list = read_data_path_txt(txt_path_list[0])
            self.support_negative_data_list = read_data_path_txt(txt_path_list[1])
            self.query_positive_data_list = read_data_path_txt(txt_path_list[2])
            self.query_negative_data_list = read_data_path_txt(txt_path_list[3])
        else:
            self.mode = 'test'
            self.test_data_list = read_data_path_txt(txt_path_list[0])

        self.seq_len = seq_len
        if self.mode == 'train':
            self.L = len(self.support_positive_data_list)
        else:
            self.L = len(self.test_data_list)
        with open('file2class.pkl', 'rb') as f:
            data2label = pickle.load(f)
        self.data2label = data2label

    def __getitem__(self, index):
        if self.mode == 'test':
            # 说明是测试阶段，只读一个
            return self.load_data(index, self.data_list[0])
        else:
            # 说明是训练阶段，同时读取support set和query set中的数据，并返回一个episode
            support_positive_tuple = self.load_data(index, self.support_positive_data_list)  # (data, label)
            support_negative_tuple = self.load_data(index, self.support_negative_data_list)  # for data
            query_positive_tuple = self.load_data(index, self.query_positive_data_list)  # (80, 1024)
            query_negative_tuple = self.load_data(index, self.query_negative_data_list)
            return support_positive_tuple, support_negative_tuple, query_positive_tuple, query_negative_tuple

    def load_data(self, index, data_path_list):
        src_path = data_path_list[index]
        src_mel = segment(np.load(str(src_path)), seglen=self.seq_len)  # 读取的时候记得把path()对象变成字符串
        data_id = src_path.parts[-1][:-4]
        label = torch.tensor(self.data2label[data_id]).long()
        return torch.FloatTensor(src_mel), label

    def __len__(self):
        return self.L
        pass


def extract_classes():
    csv_list = [r'D:\毕设相关\2018\训练集原始数据\BirdVoxDCASE20k_csvpublic.csv',
                r'D:\毕设相关\2018\训练集原始数据\ff1010bird_metadata_2018.csv',
                r'D:\毕设相关\2018\训练集原始数据\warblrb10k_public_metadata_2018.csv']

    # 将文件名完整输入class
    # data_list = []
    # with open(txt_file, encoding='utf-8') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         data_list.append(line.split('\\')[-1][:-5])

    file2class = {}
    for csv in csv_list:
        class_pd = pd.read_csv(csv)
        itemids = class_pd['itemid']
        hasbirds = class_pd['hasbird']
        for i, itemid in enumerate(itemids):
            file2class[str(itemid)] = int(hasbirds[i])

    with open('file2class.pkl', 'wb') as f:
        pickle.dump(file2class, f)

    return file2class


def generate_pairs_scripts(save_log_dir, file2class):

    if save_log_dir.exists() == False:
        print("表单的保存目录不存在！")
        exit()

    records = []
    class2file = {}

    positive_list = []
    negative_list = []

    with open('Train_Scp.txt', encoding='utf-8', mode='r') as f:
        lines = f.readlines()
        for line in lines:
            data_id = line.split('\\')[-1].strip('\n').strip('.npy')
            records.append(line)
            if file2class[data_id] == 0:
                negative_list.append(line)
            else:
                positive_list.append(line)

    random.shuffle(positive_list)
    random.shuffle(negative_list)
    class2file[0] = negative_list
    class2file[1] = positive_list

    # pivot = 12288
    pivot = 12288
    print(pivot/2)
    support_positive_list = positive_list[:int((pivot/2))]
    support_negative_list = negative_list[:int((pivot/2))]
    query_positive_list = positive_list[int((pivot/2)):pivot]
    query_negative_list = negative_list[int((pivot/2)):pivot]

    test_list = positive_list[pivot:] + negative_list[pivot:]
    print(len(test_list))
    # 写入训练集中的support_set
    with open((save_log_dir / "support_positive.txt"), 'a', encoding='utf-8') as f:
        ## './Experiments/vX/train.txt'
        for support_positive_data in support_positive_list:
            f.write(str(support_positive_data))

    with open((save_log_dir / "support_negative.txt"), 'a', encoding='utf-8') as f:
        ## './Experiments/vX/train.txt'
        for support_negative_data in support_negative_list:
            f.write(str(support_negative_data))

    # 写入训练集中的query_set
    with open((save_log_dir / "query_positive.txt"), 'a', encoding='utf-8') as f:
        for query_positive_data in query_positive_list:
            f.write(str(support_positive_data))
    with open((save_log_dir / "query_negative.txt"), 'a', encoding='utf-8') as f:
        for query_negative_data in query_negative_list:
            f.write(str(query_negative_data))


    # 写入测试集
    with open((save_log_dir / "test.txt"),'a', encoding='utf-8') as f:
        ## './Experiments/vX/test.txt'
        for test_data in test_list:
            f.write(str(test_data))

    print("*"*30 + "写入 数据集 的 训练测试 表单完毕" + "*"*30)


if __name__ == '__main__':
    with open('file2class.pkl', 'rb') as f:
        data2label = pickle.load(f)
    generate_pairs_scripts(Path("Experiments/v0"), data2label)

