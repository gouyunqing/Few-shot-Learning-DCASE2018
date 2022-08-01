import torch
from pathlib import Path
import pickle


class Config:
    def __init__(self):
        ################ preprocess  ###################################
        self.wav_datadir_name = r'D:\毕设相关\2018\训练集原始数据'  ### 原始数据集
        self.feature_dir_name = r'2018年数据集三合一_FSL'  ## 目标文件夹路径
        self.trim_db = 20  # 静音消除参数
        self.n_fft = 1024  # 提取出 513维度的傅里叶谱，再转为80维度 melspec
        self.win_length = 1024  # 帧长
        self.hop_length = 256  # 帧移
        self.sample_rate = 22050
        self.f_min = 0
        self.f_max = 22050  ## 谱（还未取对数的时候）中的 数值通常最大值设为 采样率的一半
        self.n_mels = 80

        ################ Trainer  ###################################
        self.total_iters = 1000  ## 总共训练步骤
        self.epoch_num = 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_save_every = 200  ## 每隔200次 保存一次模型
        self.lr_update_every = 200
        self.eval_every = 50

        ################ dataset / loader  ###################################
        self.train_ratio = 0.9  ## 切分训练集、测试集的比例
        self.mel_seglen = 1024  ###  训练时，谱被padding的长度
        self.min_train_mellen = 120  ### 生成表单中，所含有的 pairs的最大melspec长度。
        self.batchsize_train = 16  ## 训练的batchsize

        #################  optimizer  ##############################

        self.lr_start = 2e-5  # 初始学习率

        self.beta1 = 0.9
        self.beta2 = 0.999
        self.amsgrad = True
        self.weight_decay = 0.0001
        self.grad_norm = 3  # 梯度剪裁
        self.is_lr_decay = False  ## 训练过程中学习率是否下降

        ####################### model params #########################################
        self.class_num = 2  ## 数据集的分类数量。

        ################################################################
        ## Experiment File dir
        self.epdir = Path('./Experiments')
        ## dirs create
        self.ep_version = None
        self.ep_version_dir = None
        self.model_savedir = None
        ## files create
        self.hp_filepath = None
        self.ep_logfilepath = None
        self.ep_logfilepath_eval = None
        ################################################################

    def set_experiment(self,version='v0'):
        ## dirs create
        self.ep_version = version
        self.ep_version_dir = self.epdir / self.ep_version
        self.model_savedir = self.ep_version_dir / 'checkpoints_{}'.format(version)
        self.conversion_dir = self.ep_version_dir / 'conversion_result_{}'.format(version)
        ## files create
        self.hp_filepath = self.ep_version_dir.joinpath('hparams_{}.pickle'.format(version))
        self.ep_logfilepath = self.ep_version_dir / 'logs_{}.txt'.format(version)
        self.ep_logfilepath_eval = self.ep_version_dir / 'logs_eval_{}.txt'.format(version)


###  新的参数文件的生成.
def boot_a_new_experiment_hp(epversion,tt_iters=None,mel_segm_len=None,train_batchsize=None,use_mel_dir=None,
                             start_lr=None):
    config = Config()
    config.set_experiment(epversion) ## 设定 实验版本
    if tt_iters != None:
        config.total_iters = tt_iters
    if mel_segm_len != None:
        config.mel_seglen = mel_segm_len
    if train_batchsize != None:
        config.batchsize_train = train_batchsize
    if use_mel_dir != None:
        config.use_meldatadir = use_mel_dir
    if start_lr != None:
        config.lr_start = start_lr
    return config


def boot_a_new_experiment(epversion, tt_iters = None, mel_segm_len = None, train_batchsize = None, use_mel_dir = None,
    start_lr = None):

    hp = boot_a_new_experiment_hp(epversion, tt_iters, mel_segm_len , train_batchsize , use_mel_dir ,
    start_lr)
    ## 创建实验文件夹,
    hp.ep_version_dir.mkdir(parents=True, exist_ok=True) ## Experiment/v1/
    hp.model_savedir.mkdir(parents=True, exist_ok=True)  ## Experiment/v1/checkpoints
    ## 存储参数文件本身
    with open(hp.hp_filepath.resolve(),'wb') as hpf:
        pickle.dump(hp,hpf)
    return hp

    pass