import pickle
import torch
import numpy as np
import random
from Create_Hparams import boot_a_new_experiment
from build_dataset import generate_pairs_scripts
from trainer import Trainer
import torch.nn as nn
from build_dataset import MeldataSet
from model import Classifier


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__=="__main__":
    ##################################################################

    same_seeds(2021)# 随机数种子
    ## 开启一组实验文件
    ### 每次希望修改参数，然后run新的实验的时候，只需要把下面的 ver改变，
    ## 然后 运行main.py
    ## 注意  ，不要 已经运行过一次 "v1"，又运行一次 ver=v1 的main.py ,会出问题。
    ver = 'v0'
    vhp = boot_a_new_experiment(epversion=ver,mel_segm_len=1024,tt_iters=1500,train_batchsize=12,start_lr=0.0001,
                                )
    # generate_pairs_scripts('meldata_22k_trimed', ## 指定所使用的特征数据文件夹。
    #                        vhp.ep_version_dir,## 指定表单保存路径
    #                       vhp,
    #                       )
    ## 训练 ,从刚才生成的pickle读取 hp
    hp_file_path = str(vhp.hp_filepath)
    loaded_hp = None
    with open(hp_file_path,'rb') as f2:
        loaded_hp=pickle.load(f2)
    print("seg mel len:",loaded_hp.mel_seglen)

    t = Trainer(loaded_hp)
    # t.train_by_epoch()

    model = torch.load(r'D:\毕设相关\第一次尝试\Experiments\v0\checkpoints_v0\019999.pth')
    model.to('cpu')
    TestmeldataLoader = MeldataSet(scp_dir=r'D:\毕设相关\第一次尝试\Experiments\v0\test.txt',
                                     seglen=1024
                                     )
    pred_list = []
    label_list = []

    correct_num = 0
    with torch.no_grad():  ## 测试的过程中不需要计算 梯度。
        for batch in TestmeldataLoader:
            mels, labels = [b for b in batch]  # mels:[B,80,256] labels:[B]
            mels = mels.unsqueeze(0)
            mels = mels.unsqueeze(1).permute(0, 1, 3, 2)
            pred_prob = model(mels)  ## 输出分类概率 [B,num_class]
            batch_loss = nn.CrossEntropyLoss(pred_prob, labels)
            pred_index = torch.max(pred_prob, dim=1)[1][0]  # 求 最大概率的下标
            pred_list.append(pred_index)
            label_list.append(labels)
            batch_acc_num = (pred_index == labels).sum()
            correct_num += batch_acc_num
        batch_accuracy = correct_num / 2924  ## 准确率 = 判对数量 / 总test语音数量
        print(batch_accuracy)

    print(pred_list)
    print(label_list)



