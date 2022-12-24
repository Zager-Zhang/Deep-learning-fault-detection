#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging
import os
import time
import warnings
import torch
from torch import nn
from torch import optim
import models
from torch.utils.tensorboard import SummaryWriter


class train_utils(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer
        :return:
        """
        args = self.args

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        # Load the datasets
        if args.processing_type == 'O_A':
            from CNN_Datasets.O_A import datasets
            Dataset = getattr(datasets, args.data_name)
        elif args.processing_type == 'R_A':
            from CNN_Datasets.R_A import datasets
            Dataset = getattr(datasets, args.data_name)
        elif args.processing_type == 'R_NA':
            from CNN_Datasets.R_NA import datasets
            Dataset = getattr(datasets, args.data_name)
        else:
            raise Exception("processing type not implement")

        print(Dataset)

        self.datasets = {}
        self.datasets = {}

        self.datasets['train'], self.datasets['val'] = Dataset(args.data_dir, args.normlizetype).data_preprare()

        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x == 'train' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False))
                            for x in ['train', 'val']}
        # Define the model
        self.model = getattr(models, args.model_name)(in_channel=Dataset.inputchannel, out_channel=Dataset.num_classes)
        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Define the optimizer
        if args.opt == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr,
                                        weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")

        # Define the learning rate decay
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")

        # Load the checkpoint
        self.start_epoch = 0

        # Invert the model and define the loss
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, writer):
        """
        Training process
        :return:
        """
        args = self.args

        step = 0
        best_acc = 0.0
        batch_count = 0
        batch_loss = 0.0
        batch_acc = 0
        step_start = time.time()

        for epoch in range(self.start_epoch, args.max_epoch):

            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-' * 5)
            # Update the learning rate
            if self.lr_scheduler is not None:
                # self.lr_scheduler.step(epoch)
                logging.info('current lr: {}'.format(self.lr_scheduler.get_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))

            # Each epoch has a training and val phase
            for phase in ['train', 'val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0

                # 循环相关的变量 为了计算精确率、召回率、误报率、漏检率
                epoch_normal_correct = 0
                epoch_failure_correct = 0
                epoch_normal_incorrect = 0
                epoch_failure_incorrect = 0

                epoch_precision = 0.0  # 精确率
                epoch_recall = 0.0  # 召回率
                epoch_false_alarm = 0.0  # 误报率
                epoch_miss_rate = 0.0  # 漏报率

                # Set model to train mode or test mode
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # 四种数据初始化
                    batch_normal_correct = 0
                    batch_failure_correct = 0
                    batch_normal_incorrect = 0
                    batch_failure_incorrect = 0

                    # Do the learning process, in val, we do not care about the gradient for relaxing
                    with torch.set_grad_enabled(phase == 'train'):

                        # forward
                        logits = self.model(inputs)
                        loss = self.criterion(logits, labels)  # 交叉熵损失函数
                        pred = logits.argmax(dim=1)
                        correct = torch.eq(pred, labels).float().sum().item()
                        loss_temp = loss.item() * inputs.size(0)
                        epoch_loss += loss_temp
                        epoch_acc += correct

                        # 计算四种数据
                        batch_contrast = torch.eq(pred, labels)
                        for i in range(len(batch_contrast)):
                            if batch_contrast[i]:  # 判断正确
                                if labels[i].item() == 0:
                                    batch_normal_correct += 1  # 正常样本判断正确 TN
                                else:
                                    batch_failure_correct += 1  # 故障样本判断正确 TP
                            else:
                                if labels[i].item() == 0:
                                    batch_normal_incorrect += 1  # 正常样本判断错误 FN
                                else:
                                    batch_failure_incorrect += 1  # 故障样本判断错误 FP

                        epoch_normal_correct += batch_normal_correct
                        epoch_failure_correct += batch_failure_correct
                        epoch_normal_incorrect += batch_normal_incorrect
                        epoch_failure_incorrect += batch_failure_incorrect

                        # Calculate the training information
                        if phase == 'train':
                            # backward
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            batch_loss += loss_temp
                            batch_acc += correct
                            batch_count += inputs.size(0)

                            # Print the training information
                            if step % args.print_step == 0:
                                batch_loss = batch_loss / batch_count
                                batch_acc = batch_acc / batch_count
                                temp_time = time.time()
                                train_time = temp_time - step_start
                                step_start = temp_time
                                batch_time = train_time / args.print_step if step != 0 else train_time
                                sample_per_sec = 1.0 * batch_count / train_time
                                logging.info('Epoch: {} [{}/{}], Train Loss: {:.4f} Train Acc: {:.4f},'
                                             '{:.1f} examples/sec {:.2f} sec/batch'.format(
                                    epoch, batch_idx * len(inputs), len(self.dataloaders[phase].dataset),
                                    batch_loss, batch_acc, sample_per_sec, batch_time
                                ))
                                batch_acc = 0
                                batch_loss = 0.0
                                batch_count = 0
                            step += 1

                # Print the train and val information via each epoch
                epoch_loss = epoch_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = epoch_acc / len(self.dataloaders[phase].dataset)

                # 指标计算
                # 正确识别成故障的比例
                TN = epoch_normal_correct
                TP = epoch_failure_correct
                FN = epoch_normal_incorrect
                FP = epoch_failure_incorrect
                if TP + FP:
                    epoch_precision = TP / (TP + FP)
                if TP + FN:
                    epoch_recall = TP / (TP + FN)
                if TN + FP:
                    epoch_false_alarm = FP / (TN + FP)
                if TP + FN:
                    epoch_miss_rate = FN / (TP + FN)
                if epoch_precision + epoch_recall:
                    F1 = 2 * epoch_precision * epoch_recall / (epoch_precision + epoch_recall)

                # 将每次循环的loss和acc放入tensorboard
                writer.add_scalar(f"epoch_loss_{phase}", epoch_loss, epoch + 1)
                writer.add_scalar(f"epoch_acc_{phase}", epoch_acc, epoch + 1)
                writer.add_scalar(f"epoch_precision_{phase}", epoch_precision, epoch + 1)
                writer.add_scalar(f"epoch_recall_{phase}", epoch_recall, epoch + 1)
                writer.add_scalar(f"epoch_false_alarm_{phase}", epoch_false_alarm, epoch + 1)
                writer.add_scalar(f"epoch_miss_rate_{phase}", epoch_miss_rate, epoch + 1)
                writer.add_scalar(f"epoch_F1_{phase}", F1, epoch + 1)

                logging.info('Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, Cost {:.4f} sec'.format(
                    epoch, phase, epoch_loss, phase, epoch_acc, time.time() - epoch_start
                ))

                # save the model
                if phase == 'val':
                    # save the checkpoint for other learning
                    model_state_dic = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
                    # save the best model according to the val accuracy
                    if epoch_acc > best_acc or epoch > args.max_epoch - 2:
                        best_acc = epoch_acc
                        logging.info("save best model epoch {}, acc {:.4f}".format(epoch, epoch_acc))
                        torch.save(model_state_dic,
                                   os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)))

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
