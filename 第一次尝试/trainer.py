import torch
import torch.nn as nn
import pickle ## 帮助保存参数
from sklearn.metrics import accuracy_score ###  帮助算 分类准确率
from torch.utils.data import DataLoader  ### 加载器

from build_dataset import MeldataSet  ## 读取数据集类
from Create_Hparams import Create_Train_Hparams  ## 参数控制
from model import Classifier  ## 模型


class Trainer(object):

    def __init__(self, hp: Create_Train_Hparams):
        super(Trainer, self).__init__()

        self.hp = hp
        self.device = hp.device
        self.set_trainer_configuration()
        self.cls_nums = self.hp.speaker_nums ##

        pass

    def prepareDataloader(self):
        ## train meldata set
        self.meldataSet = MeldataSet(scp_dir=self.hp.ep_version_dir / "train.txt",
                                     seglen=self.hp.mel_seglen
                                     )
        self.DataNum = self.meldataSet.__len__()
        self.meldataLoader = DataLoader(self.meldataSet,
                                        batch_size=self.hp.batchsize_train,
                                        shuffle=True,
                                        drop_last=True)
        ## test meldata set ##
        self.TestmeldataSet = MeldataSet(scp_dir=self.hp.ep_version_dir / "test.txt",
                                     seglen=self.hp.mel_seglen
                                     )
        self.TestDataNum = self.TestmeldataSet.__len__()
        self.TestmeldataLoader = DataLoader(self.TestmeldataSet,
                                        batch_size=1,
                                        shuffle=False,
                                        drop_last=True)


    def build_models(self):

        self.current_lr = self.hp.lr_start
        self.model = Classifier(input_size=self.hp.n_mels,hidden_size=128,num_layers=2,num_classes=self.hp.speaker_nums)
        self.loss_func = nn.CrossEntropyLoss() ## 分类最常用的交叉熵概率
        self.optimizer = torch.optim.Adam(self.model.parameters(),self.current_lr,
                                            betas=(self.hp.beta1, self.hp.beta2))
        self.print_network(self.model, 'Model')

        self.model.to(self.hp.device)## 将模型放到gpu

    def set_trainer_configuration(self):
        ## 训练前的一系列初始化
        self.epoch_num = 0 ## 在迭代第几次数据集
        self.current_iter = 0  ## 当前在迭代第几次batch
        self.init_logger()    ## 创建 日志文件
        self.prepareDataloader()  ## 创建数据集
        self.build_models()  ## 创建模型
        # Step size.
        self.model_save_every = self.hp.model_save_every
        self.lr_update_every = self.hp.lr_update_every



        pass

    ##########  一些工具方法  ########################################################
    def init_logger(self):
        self.logFileName = str(self.hp.ep_logfilepath)
        if self.hp.ep_logfilepath.exists == True:
            self.hp.ep_logfilepath.exists.unlink()  ## 删除已经存在的log文件 (方便重启实验)
        self.logFileName_eval = str(self.hp.ep_logfilepath_eval)
        if self.hp.ep_logfilepath_eval.exists == True:
            self.hp.ep_logfilepath_eval.exists.unlink()  ## 删除已经存在的log文件 (方便重启实验)

        ## 存储training log文件
        with open(self.logFileName, 'a', encoding='utf-8') as wf:
            for k, v in self.hp.__dict__.items():
                wf.write("{} : {}\n".format(k, v))
            wf.write('-' * 50 + "Experiment & Hparams Created" + "-" * 50 + "\n")
            wf.write("*" * 100 + "\n")
            wf.close()

    ## 该方法将一个字典  ，按kv的顺序，写入一行到 log.txt
    def write_line2log(self, log_dict: dict, filedir, isprint: True):
        strp = ''
        with open(filedir, 'a', encoding='utf-8') as f:
            for key, value in log_dict.items():
                witem = '{}'.format(key) + ':{},'.format(value)
                strp += witem
            f.write(strp)
            #f.write('当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
            f.write('\n')
        if isprint:
            print(strp)
        pass

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print("Model {},the number of parameters: {}".format(name, num_params))

    def update_lr(self, lr):
        """Decay learning rates of the model."""

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.optimizer.zero_grad()



    def to_gpu(self, batch):
        return [b.to(self.device) for b in batch]

    def save_model(self, i):
        pdict = {"model":self.model.state_dict(),
                 }
        path = self.hp.model_savedir / "{:06}.pth".format(i)
        torch.save(pdict, str(path))
        print("---------------- model saved ------------------- ")

    def check_stop(self):
        ## 中断训练,保存模型和loss字典
        if self.current_iter > self.hp.total_iters:
            self.save_model(self.current_iter)
            savingpath = str(self.hp.ep_version_dir / 'running_loss_{}.pickle'.format(self.hp.ep_version))
            with open(savingpath, 'wb') as f1:
                pickle.dump(self.loss_log_dict, f1)
            print("*************** Training End *******************")
            exit()

    #################################################################################################
    def test_accuracy(self):
        ###  使用“test.txt"中的数据计算准确率。
        with torch.no_grad(): ## 测试的过程中不需要计算 梯度。
            for batch in self.TestmeldataLoader:
                mels, labels = self.to_gpu(batch)  # mels:[B,80,256] labels:[B]
                mels = mels.unsqueeze(1).permute(0, 1, 3, 2)
                pred_prob = self.model(mels)  ## 输出分类概率 [B,num_class]
                batch_loss = self.loss_func(pred_prob, labels)
                pred_index = torch.max(pred_prob, dim=1)[1]  # 求 最大概率的下标
                batch_acc_num = (pred_index == labels).sum()
            batch_accuracy = batch_acc_num / self.TestDataNum ## 准确率 = 判对数量 / 总test语音数量

            losse_curves  = {"test_step--":"",
                            "epoch":self.epoch_num,
                             "steps":self.current_iter,
                            "loss":batch_loss.item(),
                             "acc":batch_accuracy}
            self.write_line2log(losse_curves, self.hp.ep_logfilepath_eval, isprint=True)

    def train_a_epoch(self):
        for batch in self.meldataLoader:
            self.current_iter += 1
            self.epoch_num = self.current_iter // (self.DataNum // self.hp.batchsize_train)
            self.check_stop() ## 判断停止的条件。
            mels,labels = self.to_gpu(batch) # mels:[B,80,256] labels:[B]
            mels = mels.unsqueeze(1).permute(0,1,3,2)
            pred_prob = self.model(mels) ## 输出分类概率 [B,num_class]
            batch_loss = self.loss_func(pred_prob,labels)
            pred_index = torch.max(pred_prob, dim=1)[1]  # 求 最大概率的下标
            batch_accuracy = accuracy_score(labels.cpu(), pred_index.cpu())  # 计算准确率

            self.reset_grad()        ## 清空计算图中的梯度
            batch_loss.backward()    ## 计算 计算图每个参数的梯度
            self.optimizer.step()     ## 更新每个参数。

            losse_curves  = {"train_step--":"",
                             "epoch":self.epoch_num,
                             "steps":self.current_iter,
                            "loss":batch_loss.item(),
                             "acc":batch_accuracy,
                             }
            ## 在训练第一步完成后，建立一个字典保存losses {"loss name” : [loss_data ]}
            if self.current_iter == 1:
                print("create loss dict")
                self.loss_log_dict = {}
                for k, v in losse_curves.items():
                    self.loss_log_dict[k] = []
                print("loss dict created")
            ##############################################################################

            #######  loss save ######################################################
            for k, v in self.loss_log_dict.items():
                self.loss_log_dict[k].append(losse_curves[k])  # 把每batch的loss数据加入到 loss curves中
            self.write_line2log(losse_curves, self.hp.ep_logfilepath, isprint=True)
            #########################################################################
        ########################################### 其他东西 ##########################################
            if (self.current_iter + 1) % self.hp.eval_every == 0:
                self.test_accuracy()  ## 每 几步验证一次准确率。

            ## 模型保存
            if (self.current_iter + 1) % self.hp.model_save_every == 0:
                self.save_model(self.current_iter)

        pass

    def train_by_epoch(self):
        """
        Main training loop
        """
        print('------------- BEGIN TRAINING LOOP ----------------')
        while (1):
            self.train_a_epoch()


if __name__ == "__main__":


    # ### pytorch 一个普适的训练代码基本框架
    # # 定义数据集
    # dataset = None
    # dataloader = DataLoader(dataset,batch_size=5)
    # # 定义 模型、优化器、损失函数
    # model  = Classifier()
    # lossfunction = None
    # optimizer = None
    # # 定义 训练 循环
    # for batch in dataloader:
    #     inputs,labels = batch
    #     outputs = model(inputs)
    #     loss = lossfunction(inputs,outputs)
    #     optimizer.zero_grad()   # 计算图梯度清空。
    #     loss.backward()   ## 利用 损失值，计算每个参数的梯度
    #     optimizer.step ###  利用梯度，更新网络参数


    pass
























