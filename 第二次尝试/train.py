import torch
import torch.nn as nn
import pickle ## 帮助保存参数
from sklearn.metrics import accuracy_score ###  帮助算 分类准确率
from torch.utils.data import DataLoader  ### 加载器
from torch.optim import Adam

from load_data import MelDataset  ## 读取数据集类
from Config import Config  ## 参数控制
from model import PrototypicalNetwork  ## 模型
from tqdm import tqdm

import numpy as np
from sklearn.metrics import f1_score


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def metrics(pred, true):
    pred = np.array(pred).reshape(-1)
    true = np.array(true).reshape(-1)
    # acc
    acc = np.mean((pred == true))
    # f_score
    f_score = f1_score(true, pred, average='macro')
    return acc, f_score


class Train:
    def __init__(self, config, train_dataset, eval_dataset):
        self.config = config
        self.model = PrototypicalNetwork(input_size=self.config.n_mels,
                                         hidden_size=128,
                                         num_layers=2,
                                         num_classes=self.config.class_num,
                                         config=self.config).to(config.device)
        self.optimizer = Adam(self.model.parameters(), self.config.lr_start, betas=(self.config.beta1, self.config.beta2))
        self.criterion = nn.CrossEntropyLoss().to(self.config.device)
        self.dataloader = DataLoader(dataset=train_dataset,
                                     batch_size=self.config.batchsize_train,
                                     shuffle=True,
                                     drop_last=True)
        self.eval_dataloader = DataLoader(dataset=eval_dataset,
                                     batch_size=self.config.batchsize_train,
                                     shuffle=True,
                                     drop_last=True)
        self.total_epoch = config.epoch_num

    def to_gpu(self, batch):
        data = batch[0]
        label = batch[1]
        return data.to(self.config.device), label.to(self.config.device)

    def eval(self, curr_epoch):
        self.model.eval()
        val_losses, accuracy, f_score = [AverageMeter() for i in range(3)]
        criterion = self.criterion
        with torch.no_grad():
            for step, episode in enumerate(tqdm(self.dataloader, desc='Evaluating model')):
                support_positive_data, _ = self.to_gpu(episode[0])
                support_negative_data, _ = self.to_gpu(episode[1])
                query_positive_data, _ = self.to_gpu(episode[2])
                query_negative_data, _ = self.to_gpu(episode[3])
                x_support = torch.stack((support_negative_data, support_positive_data), dim=1).permute(0, 1, 3,
                                                                                                       2)  # (k=batch_size, n=2, 80, 1024)
                x_query = torch.stack((query_negative_data, query_positive_data), dim=1).permute(0, 1, 3, 2)
                n = self.config.class_num
                q = self.config.batchsize_train

                y_query = torch.arange(0, n, requires_grad=False).view(n, 1).expand(n, q).reshape(n * q, ).to(
                    self.config.device)
                # infer
                output = self.model(x_support, x_query).to(self.config.device)
                loss = criterion(output, y_query).item()
                val_losses.update(loss)
                # metrics of acc and f_score
                pred_ids = torch.argmax(output, dim=-1).cpu().numpy()
                y_query = y_query.cpu().numpy()
                acc, f_s = metrics(pred_ids, y_query)
                accuracy.update(acc)
                f_score.update(f_s)
                accuracy.update(acc)
                f_score.update(f_s)
            print(
                f'Epoch: {curr_epoch} evaluation results: Val_loss: {val_losses.avg}, mAP: {accuracy.avg}, F_score: {f_score.avg}')
        return val_losses.avg, accuracy.avg, f_score.avg

    def train_one_epoch(self, curr_epoch):
        self.model.train()
        train_losses = AverageMeter()
        criterion = self.criterion
        epoch_iterator = tqdm(self.dataloader,
                              desc="Training [epoch X/X | episode X/X] (loss=X.X | lr=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)
        for step, episode in enumerate(epoch_iterator):
            support_positive_data, _ = self.to_gpu(episode[0])
            support_negative_data, _ = self.to_gpu(episode[1])
            query_positive_data, _ = self.to_gpu(episode[2])
            query_negative_data, _ = self.to_gpu(episode[3])
            x_support = torch.stack((support_negative_data, support_positive_data), dim=1).permute(0,1,3,2)  # (k=batch_size, n=2, 80, 1024)
            x_query = torch.stack((query_negative_data, query_positive_data), dim=1).permute(0,1,3,2)
            n = self.config.class_num
            q = self.config.batchsize_train

            y_query = torch.arange(0, n, requires_grad=False).view(n, 1).expand(n, q).reshape(n * q, ).to(self.config.device)

            # training one episode
            self.optimizer.zero_grad()
            output = self.model(x_support, x_query).to(self.config.device)
            loss = criterion(output, y_query)
            loss.backward()
            self.optimizer.step()

            # log
            train_losses.update(loss.item())
            epoch_iterator.set_description(
                "Training [epoch %d/%d | episode %d/%d] | (loss=%2.5f | lr=%f)" %
                (curr_epoch, self.config.epoch_num, step + 1, len(epoch_iterator), loss.item(), 1)
            )
        return train_losses.avg

    def train(self):
        for curr_epoch in range(1, self.total_epoch+1):
            train_loss = self.train_one_epoch(curr_epoch)
            val_loss, accuracy, f_score = self.eval(curr_epoch)
            print(accuracy)


if __name__ == '__main__':
    config = Config()
    txt_list = [r'D:\毕设相关\第二次尝试\Experiments\v0\support_positive.txt',
                r'D:\毕设相关\第二次尝试\Experiments\v0\support_negative.txt',
                r'D:\毕设相关\第二次尝试\Experiments\v0\query_positive.txt',
                r'D:\毕设相关\第二次尝试\Experiments\v0\query_negative.txt',]
    test_txt_list = [r'D:\毕设相关\第二次尝试\Experiments\v0\test.txt']
    train_dataset = MelDataset(txt_list, 1024)
    eval_dataset = MelDataset(test_txt_list, 1024)

    trainer = Train(config=config, train_dataset=train_dataset, eval_dataset=eval_dataset)
    trainer.train()
