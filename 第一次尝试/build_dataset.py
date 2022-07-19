import torch
import numpy as np
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from Create_Hparams import Create_Train_Hparams
import pickle
import pandas as pd


## 下面两个函数 ，实现，将一个二维矩阵 补零到 指定长度。 （补一列一列的零）. 如果超过 指定的seglen，则切掉多余的。
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
        y = x[:,r:r+seglen]
    return y


class MeldataSet(Dataset):

    def __init__(self, scp_dir, seglen):
        self.scripts = []
        self.seglen = seglen
        with open(scp_dir,encoding='utf-8') as f :
            for l in f.readlines():
                self.scripts.append(Path(l.strip('\n')))
        self.L = len((self.scripts))
        with open('file2class.pkl', 'rb') as f:
            data2label = pickle.load(f)
        self.data2label = data2label

        # 建立语音标签的 查找表
        ## 对于 一个说话人分类数据集， 给定任意一个说话人编号 集合 {001 002 005 008 ...}
        ## 为了 crossentroy 损失函数的要求，则需要一个 下标映射查找表。
        self.speaker_names_set = list(set( [ p.parts[1] for p in self.scripts  ]))
        print("speakernames_set:{}".format(self.speaker_names_set))
        # p.parts[1] 即为路径的第二个元素，\meldata_22k_trimed\0003\001.npy
        # 把所有的路径 的第二个元素 拿出来，取个集合，就得到了 【0001,0003 ,0007,0012,0015】
        ## 每条语音的真实标签类别就根据这个表去查找。
        ##################

        pass

    def __getitem__(self, index):

        src_path = self.scripts[index]
        src_mel = segment(np.load(str(src_path)), seglen=self.seglen)  # 读取的时候记得把path()对象变成字符串
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


def generate_pairs_scripts(meldatadir_name, save_log_dir, hp:Create_Train_Hparams):

    if save_log_dir.exists() == False:
        print("表单的保存目录不存在！")
        exit()

    records = []
    with open('Train_Scp.txt', encoding='utf-8', mode='r') as f:
        lines = f.readlines()
        for line in lines:
            records.append(line)

    random.shuffle(records)
    pivot = int(len(records)/10)
    train_list = records[pivot:]
    test_list = records[:pivot]

    ###  写入训练集
    with open((save_log_dir /"train.txt"),'a', encoding='utf-8') as f:
        ## './Experiments/vX/train.txt'
        for train_data in train_list:
            f.write(str(train_data))

    ###  写入测试集
    with open((save_log_dir /"test.txt"),'a', encoding='utf-8') as f:
        ## './Experiments/vX/test.txt'
        for test_data in test_list:
            f.write(str(test_data))

    print("*"*30 + "写入 数据集 的 训练测试 表单完毕" + "*"*30)

if __name__=="__main__":
    # hp = Create_Train_Hparams()
    # # generate_pairs_scripts("meldata_22k_trimed",Path("Experiments/v0"),hp)
    # meldataSet = MeldataSet(scp_dir=Path("Experiments/v0") / "train.txt",
    #                              seglen=1024
    #                              )
    # print(meldataSet[0])## getitem 方法 返回了 2个元素
    #
    # pass
    extract_classes()


