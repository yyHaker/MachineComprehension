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

from utils import squad_evaluate, ensure_dir, pad_list
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
            input_data, label = self.build_data(data)
            # if self.config["arch"]["type"] == "BiDAFMultiParasOrigin":
            #     p1, p2 = self.model(input_data)
            # else:
            p1, p2, score = self.model(input_data)
            self.optimizer.zero_grad()

            batch_size = p1.shape[0]
            max_ans_num = data.s_idxs.shape[1]
            max_p_len = input_data['paras_word'].shape[2]
            max_p_num = input_data['paras_word'].shape[1]
            match_scores = F.softmax(torch.Tensor(pad_list(label['match_scores'], pad=-1e12)).to(self.device), dim=1)

            reshape_s_idxs = data.s_idxs.reshape(-1)
            reshape_e_idxs = data.e_idxs.reshape(-1)
            reshape_match_scores = match_scores.reshape(-1)
            reshape_answer_para_idxs = data.answer_para_idxs.reshape(-1)

            # print(f'Data:{data}')
            # print(f'max_p_len:{max_p_len}')
            # print(f'reshape_s_idxs:{reshape_s_idxs}')
            # print(f'reshape_e_idxs:{reshape_e_idxs}')
            # print(f'reshape_match_scores:{reshape_match_scores}')
            # print(f'reshape_answer_para_idxs:{reshape_answer_para_idxs}')
            # print('Assert idx < max_p_len*max_p_num:')
            # print(reshape_s_idxs >= max_p_len * max_p_num)
            # print(reshape_e_idxs >= max_p_len * max_p_num)
            # print('assert answer_para_idxs < max_p_num')
            # print(reshape_answer_para_idxs >= max_p_num)

            dup_p1 = p1.unsqueeze(1).expand(-1, max_ans_num, -1).reshape(batch_size * max_ans_num, -1)
            dup_p2 = p2.unsqueeze(1).expand(-1, max_ans_num, -1).reshape(batch_size * max_ans_num, -1)
            dup_score = score.unsqueeze(1).expand(-1, max_ans_num, -1).reshape(batch_size * max_ans_num, -1)

            # print(f'p1:{p1}')
            # print(f'p2:{p2}')
            #
            # print(f'dup_p1:{dup_p1}')
            # print(f'dup_p2:{dup_p2}')
            # print(f'dup_score:{dup_score}')

            # 计算偏移
            reshape_s_idxs = reshape_s_idxs + reshape_answer_para_idxs * max_p_len
            reshape_e_idxs = reshape_e_idxs + reshape_answer_para_idxs * max_p_len
            # print('After:')
            # print(reshape_s_idxs)
            # print(reshape_e_idxs)
            # print('assert:')
            # print(reshape_s_idxs >= max_p_len*max_p_num)
            # print(reshape_e_idxs >= max_p_len * max_p_num)

            lamda = self.config["loss"]["lamda"]
            ans_span_loss = (self.loss(dup_p1, reshape_s_idxs) + self.loss(dup_p2, reshape_e_idxs)) * reshape_match_scores
            pr_loss = self.loss(dup_score, reshape_answer_para_idxs) * reshape_match_scores
            all_loss = torch.mean((1 - lamda) * ans_span_loss + lamda * pr_loss)

            all_loss.backward()
            self.optimizer.step()
            # # 验证词向量是否部分训练
            # sep_idx = self.data_loader.vocab.stoi['<sep>']
            # eop_idx = self.data_loader.vocab.stoi['<eop>']
            #
            # fix_ebd = data.q_word[0][0][:4]
            # self.logger.info('Train ebd before:')
            # self.logger.info(self.model.module.word_emb(torch.tensor([sep_idx, eop_idx], device=torch.device('cuda:0'))))
            # self.logger.info('Fix ebd before:')
            # self.logger.info(self.model.module.word_emb(fix_ebd))

            # self.logger.info('Train ebd after:')
            # self.logger.info(
            #     self.model.module.word_emb(torch.tensor([sep_idx, eop_idx], device=torch.device('cuda:0'))))
            # self.logger.info('Fix ebd after:')
            # self.logger.info(self.model.module.word_emb(fix_ebd))

            total_loss += all_loss.item() * p1.size()[0]
            if batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx,
                    len(self.data_loader.train_iter),
                    100.0 * batch_idx / len(self.data_loader.train_iter),
                    all_loss.item()))
            # add scalar to writer
            global_step = (epoch-1) * len(self.data_loader.train) + batch_idx
            self.writer.add_scalar('train_loss', all_loss.item(), global_step=global_step)

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
                # if self.config["arch"]["type"] == "BiDAFMultiParasOrigin":
                #     p1, p2 = self.model(input_data)
                # else:
                p1, p2, score = self.model(input_data)
                # get pred ans para idx
                pred_para_idx_tensor = torch.argmax(F.softmax(score, dim=1), dim=1)

                batch_size = p1.shape[0]
                max_ans_num = data.s_idxs.shape[1]
                max_p_len = input_data['paras_word'].shape[2]
                max_p_num = input_data['paras_word'].shape[1]
                match_scores = F.softmax(torch.Tensor(pad_list(label['match_scores'], pad=-1e12)).to(self.device), dim=1)

                reshape_s_idxs = data.s_idxs.reshape(-1)
                reshape_e_idxs = data.e_idxs.reshape(-1)
                reshape_match_scores = match_scores.reshape(-1)
                reshape_answer_para_idxs = data.answer_para_idxs.reshape(-1)

                dup_p1 = p1.unsqueeze(1).expand(-1, max_ans_num, -1).reshape(batch_size * max_ans_num, -1)
                dup_p2 = p2.unsqueeze(1).expand(-1, max_ans_num, -1).reshape(batch_size * max_ans_num, -1)
                dup_score = score.unsqueeze(1).expand(-1, max_ans_num, -1).reshape(batch_size * max_ans_num, -1)

                # 计算偏移
                reshape_s_idxs = reshape_s_idxs + reshape_answer_para_idxs * max_p_len
                reshape_e_idxs = reshape_e_idxs + reshape_answer_para_idxs * max_p_len

                lamda = self.config["loss"]["lamda"]
                ans_span_loss = (self.loss(dup_p1, reshape_s_idxs) + self.loss(dup_p2, reshape_e_idxs)) * reshape_match_scores
                pr_loss = self.loss(dup_score, reshape_answer_para_idxs) * reshape_match_scores
                all_loss = torch.mean((1 - lamda) * ans_span_loss + lamda * pr_loss)

                # add scalar to writer
                global_step = (epoch - 1) * len(self.data_loader.dev) + batch_idx
                self.writer.add_scalar('eval_loss', all_loss.item(), global_step=global_step)

                total_loss += all_loss.item() * p1.size()[0]

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
                    gold_para_idxs.append(data.answer_para_idxs[i].tolist())
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
        if not self.config["arch"]["type"] == "BiDAFMultiParasOrigin":
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
            'match_scores': batch.match_scores,
            'answer_para_idxs': batch.answer_para_idxs
        }
        return input_data, label

    @staticmethod
    def get_acc(pred, gold):
        right = 0.0
        for i in range(len(pred)):
            if pred[i] in gold[i]:
                right += 1
        return right / len(pred)
