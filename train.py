import os

import torch as t
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import Config
from dataset import HandWriteDataSet
from model import DigitsMobilenet


class Trainer:

    def __init__(self, dataset):
        self.config = Config()
        self.device = t.device('cuda') if t.cuda.is_available() else t.device(
            'cpu')
        self.train_loader = DataLoader(dataset,
                                       batch_size=self.config.batch_size,
                                       num_workers=0,
                                       pin_memory=True,
                                       drop_last=True,
                                       shuffle=True)
        self.val_loader = DataLoader(dataset,
                                     batch_size=self.config.batch_size,
                                     num_workers=0,
                                     pin_memory=True,
                                     drop_last=True,
                                     shuffle=True)
        self.model = DigitsMobilenet(self.config.class_num).to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = SGD(self.model.parameters(),
                             lr=self.config.lr,
                             momentum=self.config.momentum,
                             weight_decay=self.config.weights_decay,
                             nesterov=True)
        # self.lr_scheduler = CosineAnnealingWarmRestarts(self.optimizer, 10, 2)
        self.lr_scheduler = lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[i for i in range(self.config.start_epoch, self.config.epoches, 10)],
            gamma=0.5)
        self.best_acc = 0
        if self.config.pretrained is not None:
            self.load_model(self.config.pretrained, save_opt=True)
            print('Load model from %s' % self.config.pretrained)

    def train(self):
        for epoch in range(self.config.start_epoch, self.config.epoches):
            print("正在第{} 个epoch".format(epoch))
            total_loss = 0
            corrects = 0
            tbar = tqdm(self.train_loader)
            self.model.train()
            for i, (img, label) in enumerate(tbar):
                img = img.to(self.device)
                label = label.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(img)
                loss = self.criterion(pred[0], label[:, 0]) + \
                    self.criterion(pred[1], label[:, 1]) + \
                    self.criterion(pred[2], label[:, 2]) + \
                    self.criterion(pred[3], label[:, 3]) + \
                    self.criterion(pred[4], label[:, 4]) + \
                    self.criterion(pred[5], label[:, 5]) + \
                    self.criterion(pred[6], label[:, 6]) + \
                    self.criterion(pred[7], label[:, 7]) + \
                    self.criterion(pred[8], label[:, 8]) + \
                    self.criterion(pred[9], label[:, 9]) + \
                    self.criterion(pred[10], label[:, 10]) + \
                    self.criterion(pred[11], label[:, 11]) + \
                    self.criterion(pred[12], label[:, 12]) + \
                    self.criterion(pred[13], label[:, 13]) + \
                    self.criterion(pred[14], label[:, 14]) + \
                    self.criterion(pred[15], label[:, 15]) + \
                    self.criterion(pred[16], label[:, 16]) + \
                    self.criterion(pred[17], label[:, 17]) + \
                    self.criterion(pred[18], label[:, 18]) + \
                    self.criterion(pred[19], label[:, 19]) + \
                    self.criterion(pred[20], label[:, 20])
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                temp = t.stack([\
                    pred[0].argmax(1) == label[:, 0], \
                    pred[1].argmax(1) == label[:, 1],\
                    pred[2].argmax(1) == label[:, 2], \
                    pred[3].argmax(1) == label[:, 3], \
                    pred[4].argmax(1) == label[:, 4], \
                    pred[5].argmax(1) == label[:, 5], \
                    pred[6].argmax(1) == label[:, 6], \
                    pred[7].argmax(1) == label[:, 7], \
                    pred[8].argmax(1) == label[:, 8], \
                    pred[9].argmax(1) == label[:, 9], \
                    pred[10].argmax(1) == label[:, 10], \
                    pred[11].argmax(1) == label[:, 11], \
                    pred[12].argmax(1) == label[:, 12], \
                    pred[13].argmax(1) == label[:, 13], \
                    pred[14].argmax(1) == label[:, 14], \
                    pred[15].argmax(1) == label[:, 15], \
                    pred[16].argmax(1) == label[:, 16], \
                    pred[17].argmax(1) == label[:, 17], \
                    pred[18].argmax(1) == label[:, 18], \
                    pred[19].argmax(1) == label[:, 19], \
                    pred[20].argmax(1) == label[:, 20]
                ], dim=1)
                corrects += t.all(temp, dim=1).sum().item()
                if (i + 1) % self.config.print_interval == 0:
                    tbar.set_description('loss: %.6f, acc: %.3f' %
                                         (loss / (i + 1), corrects * 100 /
                                          ((i + 1) * self.config.batch_size)))
            if (epoch + 1) % self.config.eval_interval == 0:
                print('Start Evaluation')
                acc = self.eval()
                self.lr_scheduler.step()
                if acc > self.best_acc:
                    os.makedirs(self.config.checkpoints, exist_ok=True)
                    save_path = self.config.checkpoints + 'epoch-%d_acc-%.2f.pth' % (
                        epoch + 1, acc)
                    self.save_model(save_path, save_opt=True)
                    print('%s saved successfully...' % save_path)
                    self.best_acc = acc

    def eval(self):
        self.model.eval()
        corrects = 0
        with t.no_grad():
            tbar = tqdm(self.val_loader)
            for i, (img, label) in enumerate(tbar):
                img = img.to(self.device)
                label = label.to(self.device)
                pred = self.model(img)
                temp = t.stack([\
                        pred[0].argmax(1) == label[:, 0], \
                        pred[1].argmax(1) == label[:, 1], \
                        pred[2].argmax(1) == label[:, 2], \
                        pred[3].argmax(1) == label[:, 3], \
                        pred[4].argmax(1) == label[:, 4], \
                        pred[5].argmax(1) == label[:, 5], \
                        pred[6].argmax(1) == label[:, 6], \
                        pred[7].argmax(1) == label[:, 7], \
                        pred[8].argmax(1) == label[:, 8], \
                        pred[9].argmax(1) == label[:, 9], \
                        pred[10].argmax(1) == label[:, 10], \
                        pred[11].argmax(1) == label[:, 11], \
                        pred[12].argmax(1) == label[:, 12], \
                        pred[13].argmax(1) == label[:, 13], \
                        pred[14].argmax(1) == label[:, 14], \
                        pred[15].argmax(1) == label[:, 15], \
                        pred[16].argmax(1) == label[:, 16], \
                        pred[17].argmax(1) == label[:, 17], \
                        pred[18].argmax(1) == label[:, 18], \
                        pred[19].argmax(1) == label[:, 19], \
                        pred[20].argmax(1) == label[:, 20]
                    ], dim=1)
                corrects += t.all(temp, dim=1).sum().item()
                tbar.set_description('Val Acc: %.2f' %
                                     (corrects * 100 /
                                      ((i + 1) * self.config.batch_size)))
        self.model.train()
        return corrects * 100 / (len(self.val_loader) * self.config.batch_size)

    def save_model(self, save_path, save_opt=False, save_config=False):
        dicts = {'model': self.model.state_dict()}
        if save_opt:
            dicts['opt'] = self.optimizer.state_dict()
        if save_config:
            dicts['config'] = {
                s: self.config.__getattribute__(s)
                for s in dir(self.config)
                if not s.startswith('_')
            }
        t.save(dicts, save_path)

    def load_model(self, load_path, save_opt=False, save_config=False):
        dicts = t.load(load_path)
        self.model.load_state_dict(dicts['model'])
        if save_opt:
            self.optimizer.load_state_dict(dicts['opt'])
        if save_config:
            for k, v in dicts['config'].items():
                self.config.__setattr__(k, v)


if __name__ == "__main__":
    t.cuda.empty_cache()
    t.cuda.set_device(0)
    train_dataset = HandWriteDataSet("./handWriting/train.csv", train_flag=True,test_flag=False)
    trainer = Trainer(train_dataset)
    trainer.train()
