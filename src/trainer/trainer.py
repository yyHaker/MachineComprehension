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
from copy import deepcopy

from utils import *
from du_evaluation_metric import calc_score

import torch.nn.functional as F
import torch


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
            # self.logger.info(self.model.module.word_emb.weight.device)
            # self.logger.info(data.q_word[0].device)
            # self.logger.info(self.device)
            input_data, label = self.build_data(data)
            if self.config["arch"]["type"] == "BiDAFMultiParasOrigin":
                p1, p2 = self.model(input_data)
            else:
                p1, p2, score = self.model(input_data)
            # p1, p2 = self.model(data)
            self.optimizer.zero_grad()

            # use single para
            # s_idx = data.s_idx
            # e_idx = data.e_idx

            # 计算s_idx, e_idx在多个para连接时的绝对值 (for one answer)
            max_p_len = input_data['paras_word'].shape[2]
            s_idx = label['s_idx'] + label['answer_para_idx'] * max_p_len
            e_idx = label['e_idx'] + label['answer_para_idx'] * max_p_len
            # calc loss()
            lamda = self.config["loss"]["lamda"]
            if self.config["arch"]["type"] == "BiDAFMultiParasOrigin":
                loss = self.loss(p1, s_idx) + self.loss(p2, e_idx)
            else:
                loss = (1 - lamda) * (self.loss(p1, s_idx) + self.loss(p2, e_idx)) + lamda * self.loss(score,
                                                                                                       data.answer_para_idx)

            # 计算s_idx, e_idx在多个para连接时的绝对值 (for multiple answer)
            # max_p_len = input_data['paras_word'].shape[2]
            # ans_num = label['s_idxs'].shape[1]
            # batch_size = label['s_idxs'].shape[0]
            # # for every answer to calc weighted loss
            # match_scores = F.softmax(torch.Tensor(pad_list(label['match_scores'], pad=INF)).to(self.device), dim=-1)
            # ans_loss = torch.zeros(label['s_idxs'].shape[0]).to(self.device)
            # idx = 0
            # s_idx = label['s_idxs'][:, idx] + label['answer_para_idxs'][:, idx] * max_p_len
            # e_idx = label['e_idxs'][:, idx] + label['answer_para_idxs'][:, idx] * max_p_len
            #
            # target1 = torch.empty(batch_size, dtype=torch.long).random_(5).to(self.device)
            # target2 = torch.empty(batch_size, dtype=torch.long).random_(5).to(self.device)
            # ans_loss = (self.loss(p1, target1) + self.loss(p2, target2))  # [b]
            # # calc loss
            # lamda = self.config["loss"]["lamda"]
            # if self.config["arch"]["type"] == "BiDAFMultiParasOrigin":
            #     loss = torch.mean(ans_loss)
            # else:
            #     # pr_loss = torch.zeros(label['s_idxs'].shape[0]).to(self.device)
            #     loss = ans_loss
            #     # for j in range(ans_num):
            #     #     pr_loss += self.loss(score, label['answer_para_idxs'][:, j]) * match_scores[:, j]
            #     # loss = (1 - lamda) * torch.mean(ans_loss) + lamda * torch.mean(pr_loss)

            # 计算s_idx, e_idx在多个para连接时的绝对值 (for multiple answer)
            # max_p_len = input_data['paras_word'].shape[2]
            # ans_num = label['s_idxs'].shape[1]
            # batch_size = label['s_idxs'].shape[0]
            # # for every answer to calc weighted loss
            # match_scores = F.softmax(torch.Tensor(pad_list(label['match_scores'], pad=INF)).to(self.device), dim=-1).reshape(-1)
            # s_idxs = label['s_idxs'].reshape(-1) + label["answer_para_idxs"].reshape(-1) * max_p_len
            # e_idxs = label['e_idxs'].reshape(-1) + label["answer_para_idxs"].reshape(-1) * max_p_len  # (b*ans_num)
            # p1s = p1.unsqueeze(1).expand(-1, ans_num, -1).reshape(batch_size*ans_num, -1)  # (b*ans_num, max_p_len)
            # p2s = p2.unsqueeze(1).expand(-1, ans_num, -1).reshape(batch_size*ans_num, -1)  # (b*ans_num, max_p_len)
            #
            # loss = torch.mean((self.loss(p1s, s_idxs) + self.loss(p2s, e_idxs)) * match_scores)

            # calc loss
            # lamda = self.config["loss"]["lamda"]
            # # loss = (1 - lamda) * torch.mean(torch.stack(ans_losses, dim=0).sum(dim=0))
            # # lamda * torch.mean(torch.stack(pr_losses, dim=0).sum(dim=0))
            loss.backward()
            self.optimizer.step()

            total_loss = total_loss + loss.item() * p1.size()[0]
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
        self.logger.info("Training epoch {} done, avg loss: {}, ROUGE-L :{}, BLUE-4: {}".format(epoch, avg_loss,
                                                                                               result["ROUGE-L"], result["BLUE-4"]))
        self.writer.add_scalar("eval_ROUGE-L", result["ROUGE-L"], global_step=epoch * len(self.data_loader.train))
        self.writer.add_scalar("eval_BLUE-4", result["BLUE-4"], global_step=epoch * len(self.data_loader.train))
        return result

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_acc = 0.
        total_loss = 0.
        preds = []

        span_para_idxs = []
        pred_para_idxs = []
        gold_para_idxs = []

        with torch.no_grad():
            self.data_loader.eval_iter.device = self.device
            for batch_idx, data in enumerate(self.data_loader.eval_iter):
                input_data, label = self.build_data(data)
                # s_idx, e_idx, best_para_idx = self.model(data, train=False)
                if self.config["arch"]["type"] == "BiDAFMultiParasOrigin":
                    p1, p2 = self.model(input_data)
                else:
                    p1, p2, score = self.model(input_data)
                    # get pred ans para idx
                    pred_para_idx_tensor = torch.argmax(F.softmax(score, dim=1), dim=1)
                # use multi para
                # max_p_len = input_data['paras_word'].shape[2]
                # s_idx = label['s_idx'] + label['answer_para_idx'] * max_p_len
                # e_idx = label['e_idx'] + label['answer_para_idx'] * max_p_len

                # use single para
                # s_idx = data.s_idx
                # e_idx = data.e_idx

                # lamda = self.config["loss"]["lamda"]
                # # loss = self.loss(p1, s_idx) + self.loss(p2, e_idx)
                # if self.config["arch"]["type"] == "BiDAFMultiParasOrigin":
                #     loss = self.loss(p1, s_idx) + self.loss(p2, e_idx)
                # else:
                #     loss = (1 - lamda) * (self.loss(p1, s_idx) + self.loss(p2, e_idx)) + lamda * self.loss(score,
                #                                                                                            data.answer_para_idx)

                # 计算s_idx, e_idx在多个para连接时的绝对值 (for multiple answer)
                max_p_len = input_data['paras_word'].shape[2]
                ans_num = label['s_idxs'].shape[1]
                # for every answer to calc weighted loss
                match_scores = F.softmax(torch.Tensor(pad_list(label['match_scores'], pad=INF)), dim=-1)
                ans_loss = torch.zeros(label['s_idxs'].shape[0]).to(self.device)
                for idx in range(ans_num):
                    s_idx = label['s_idxs'][:, idx] + label['answer_para_idxs'][:, idx] * max_p_len
                    e_idx = label['e_idxs'][:, idx] + label['answer_para_idxs'][:, idx] * max_p_len
                    ans_loss += (self.loss(p1, s_idx) + self.loss(p2, e_idx)) * match_scores[:, idx]  # [b]
                # calc loss
                lamda = self.config["loss"]["lamda"]
                if self.config["arch"]["type"] == "BiDAFMultiParasOrigin":
                    loss = torch.mean(ans_loss)
                else:
                    pr_loss = torch.zeros(label['s_idxs'].shape[0]).to(self.device)
                    for j in range(ans_num):
                        pr_loss += self.loss(score, label['answer_para_idxs'][:, j]) * match_scores[:, j]
                    loss = (1 - lamda) * torch.mean(ans_loss) + lamda * torch.mean(pr_loss)

                # add scalar to writer
                global_step = (epoch - 1) * len(self.data_loader.dev) + batch_idx
                self.writer.add_scalar('eval_loss', loss.item(), global_step=global_step)

                total_loss += loss.item() * p1.size()[0]

                # 统计得到的answers
                # (batch, c_len, c_len)
                batch_size, c_len = p1.size()
                ls = nn.LogSoftmax(dim=1)
                # 下三角形mask矩阵(保证i<=j)
                mask = (torch.ones(c_len, c_len) * float('-inf')).to(self.device).tril(-1).unsqueeze(0).expand(batch_size, -1, -1)
                # masked (batch, c_len, c_len)
                score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
                # s_idx: [batch, c_len]
                score, s_idx = score.max(dim=1)
                # e_idx: [batch], score is for (s_idx, e_idx).
                score, e_idx = score.max(dim=1)
                s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()
                # for multiple para,  (batch,max_para_num * max_p_len)
                concat_paras_words_idx = data.paras_word[0].reshape(data.paras_word[0].shape[0], -1)
                for i in range(batch_size):
                    pred = {}
                    # get question id, answer, question
                    q_id = data.id[i]
                    answer = concat_paras_words_idx[i][s_idx[i]:e_idx[i] + 1]
                    answer = ''.join([self.data_loader.PARAS.vocab.itos[idx] for idx in answer])
                    question = data.q_word[0][i]
                    question = ''.join([self.data_loader.PARAS.vocab.itos[idx] for idx in question])
                    # get all para idx
                    span_para_idxs.append(int(s_idx[i].item() // max_p_len))
                    gold_para_idxs.append(int(data.answer_para_idx[i].item()))
                    if not self.config["arch"]["type"] == "BiDAFMultiParasOrigin":
                        pred_para_idxs.append(int(pred_para_idx_tensor[i].item()))
                    # for pred
                    pred["question_id"] = q_id
                    pred["question"] = question
                    pred["answers"] = [answer]
                    pred["question_type"] = data.question_type[i]
                    pred["yesno_answers"] = []  # not predict now
                    preds.append(pred)

        # calc para acc
        span_acc = self.get_acc(span_para_idxs, gold_para_idxs)
        if self.config["arch"]["type"] == "BiDAFMultiParas":
            # self.logger.info(pred_para_idxs[:100])
            # self.logger.info(gold_para_idxs[:100])
            # self.logger.info(span_para_idxs[:100])
            pr_acc = self.get_acc(pred_para_idxs, gold_para_idxs)
            self.logger.info('Val Epoch: {}, span acc: {:.6f}, pr acc: {:.6f}'.format(epoch, span_acc, pr_acc))
        else:
            # self.logger.info(span_para_idxs[:100])
            self.logger.info('Val Epoch: {}, span acc: {:.6f}'.format(epoch, span_acc))
        # evaluate (F1 and Rouge_L)
        predict_file = self.config["trainer"]["prediction_file"]
        ensure_dir(os.path.split(predict_file)[0])
        with codecs.open(predict_file, 'w', encoding='utf-8') as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                print("", file=f)
        # ref file
        # ref_file = self.config["trainer"]["ref_file"]
        dev_file = os.path.join(self.config["data_loader"]["args"]["data_path"],
                                self.config["data_loader"]["args"]["dev_file"])
        ref_file = dev_file
        results = calc_score(predict_file, ref_file)
        # metrics dict
        # metrics = np.array([val_loss])
        return results

    def build_data(self, batch):
        input_data = {
            'q_word': batch.q_word[0],
            'q_lens': batch.q_word[1],
            'paras_word': batch.paras_word[0],
            'paras_num': batch.paras_word[1],
            'paras_lens': batch.paras_word[2],
        }
        label = {
            's_idxs': batch.s_idxs,
            'e_idxs': batch.e_idxs,
            'answer_para_idxs': batch.answer_para_idxs,
            'match_scores': batch.match_scores
        }
        return input_data, label

    @staticmethod
    def get_acc(pred, gold):
        right = 0.0
        for i in range(len(pred)):
            if pred[i] == gold[i]:
                right += 1
        return right / len(pred)
