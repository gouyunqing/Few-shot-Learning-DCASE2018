import pickle
import torch
import numpy as np
import random
from Create_Hparams import boot_a_new_experiment
from build_dataset import generate_pairs_scripts
from trainer import Trainer
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
    t.train_by_epoch()