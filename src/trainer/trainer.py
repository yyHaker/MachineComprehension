#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: trainer.py
@time: 2019/3/9 15:45
"""
import torch
import torch.nn as nn
import numpy as np
from .base_trainer import BaseTrainer
import json
import codecs
import os

from utils import squad_evaluate, ensure_dir


class Trainer(BaseTrainer):
    """
    Trainer class.
    Note:
        Inherited from BaseTrainer.
        ------
        realize the _train_epoch method.
    """
    def __init__(self, model, loss, metrics, optimizer, resume, config,
                 data_loader, logger=None):
        """Trainer.
        :param model:
        :param loss:
        :param metrics:
        :param optimizer:
        :param resume:
        :param config:
        :param data_loader:
        :param logger:
        """
        super(Trainer, self).__init__(model, loss, metrics, optimizer, resume, config)
        # data loader
        self.data_loader = data_loader
        # if do validation
        self.do_validation = self.data_loader.eval_iter is not None

        # log step, in every epoch, every batches to log
        # self.log_step = int(np.sqrt(data_loader.train_batch_size))

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        total_loss = 0.
        # begin train
        self.data_loader.train_iter.device = self.device
        for batch_idx, data in enumerate(self.data_loader.train_iter):
            p1, p2 = self.model(data)
            self.optimizer.zero_grad()
            loss = self.loss(p1, data.s_idx) + self.loss(p2, data.e_idx)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * p1.size()[0]
            if batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx,
                    len(self.data_loader.train_iter),
                    100.0 * batch_idx / len(self.data_loader.train_iter),
                    loss.item()))
            # add scalar to writer
            global_step = (epoch-1) * len(self.data_loader.train) + batch_idx
            self.writer.add_scalar('train_loss', loss.item(), global_step=global_step)

        # if train
        avg_loss = total_loss / (len(self.data_loader.train) + 0.)
        metrics = np.array([avg_loss])
        result = {
            "train_metrics": metrics
        }
        # if evaluate
        if self.do_validation:
            result = self._valid_epoch(epoch)
        self.logger.info("Training epoch {} done, avg loss: {}, EM :{}, f1: {}".format(epoch, avg_loss,
                                                                                       result["EM"], result["f1"]))
        self.writer.add_scalar("eval_EM", result["EM"], global_step=epoch * len(self.data_loader.train))
        self.writer.add_scalar("eval_f1", result["f1"], global_step=epoch * len(self.data_loader.train))
        return {
            "EM": result["EM"],
            "f1": result["f1"]
        }

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_loss = 0.
        answers = dict()
        with torch.no_grad():
            self.data_loader.eval_iter.device = self.device
            for batch_idx, data in enumerate(self.data_loader.eval_iter):
                p1, p2 = self.model(data)
                loss = self.loss(p1, data.s_idx) + self.loss(p2, data.e_idx)

                # add scalar to writer
                global_step = (epoch - 1) * len(self.data_loader.dev) + batch_idx
                self.writer.add_scalar('eval_loss', loss.item(), global_step=global_step)

                total_loss += loss.item() * p1.size()[0]

                # 统计得到的answers
                # (batch, c_len, c_len)
                batch_size, c_len = p1.size()
                ls = nn.LogSoftmax(dim=1)
                mask = (torch.ones(c_len, c_len) * float('-inf')).to(self.device).tril(-1).unsqueeze(0).expand(batch_size, -1, -1)
                score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
                score, s_idx = score.max(dim=1)
                score, e_idx = score.max(dim=1)
                s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()

                for i in range(batch_size):
                    id = data.id[i]
                    answer = data.c_word[0][i][s_idx[i]:e_idx[i] + 1]
                    answer = ' '.join([self.data_loader.WORD.vocab.itos[idx] for idx in answer])
                    answers[id] = answer

        # calc loss
        val_loss = total_loss / (len(self.data_loader.dev) + 0.)
        self.logger.info('Val Epoch: {}, loss: {:.6f}'.format(epoch, val_loss))
        # evaluate (F1 and EM)
        predict_file = self.config["trainer"]["prediction_file"]
        ensure_dir(os.path.split(predict_file)[0])
        with codecs.open(predict_file, 'w', encoding='utf-8') as f:
            print(json.dumps(answers), file=f)
        results = squad_evaluate.main(self.config["trainer"]["dataset_file"], self.config["trainer"]["prediction_file"])
        # metrics dict
        # metrics = np.array([val_loss])
        return {
            "EM": results["exact_match"],
            "f1": results["f1"]
        }
