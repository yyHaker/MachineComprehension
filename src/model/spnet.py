#!/usr/bin/python
# coding:utf-8

"""融合para ranking和answer prediction的spnet.
@author: yyhaker
@contact: 572176750@qq.com
@file: spnet.py
@time: 2019/4/18 21:15
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.nn import LSTM, Linear
from utils import *
INF = 1e30  # 定义正无穷


class SPNet(nn.Module):
    """SPNet"""
    def __init__(self, args, pretrained, trainable_weight_idx):
        super(SPNet, self).__init__()
        self.args = args["arch"]["args"]

        # 1. word embedding layer
        # self.word_emb = nn.Embedding.from_pretrained(pretrained, freeze=True)
        self.word_emb = PartiallyTrainEmbedding(pretrained, trainable_weight_idx)

        # 2. contextual embedding layer
        self.context_lstm = LSTM(input_size=self.args["word_dim"],
                                 hidden_size=self.args["hidden_size"],
                                 batch_first=True,
                                 num_layers=1,
                                 bidirectional=True,
                                 dropout=self.args["dropout"])
        # 3. co-attention layer
        self.att_weight_co = Linear(self.args["hidden_size"]*2, 1)

        # 4. fuse layer1 to combine
        self.fuse_m_layer = Linear(self.args["hidden_size"]*8, self.args["hidden_size"]*2)
        self.fuse_g_layer = Linear(self.args["hidden_size"]*8, self.args["hidden_size"]*2)

        # 5.  self-attention layer
        self.att_weight_s = Linear(1, 1)

        # 6. fuse layer2 to combine

        # 7. question self-align layer
        self.align_weight_q = Linear(self.args["hidden_size"]*2, 1)

        # 8. para self-align layer
        self.align_weight_p = Linear(self.args["hidden_size"]*2, 1)

        # 9. relevance score(for task 1)
        self.score_weight_qp = nn.Bilinear(self.args["hidden_size"]*2, self.args["hidden_size"]*2, 1)

        # 9. shared LSTM layer (for task2)
        self.shared_lstm = LSTM(input_size=self.args["hidden_size"]*2,
                                hidden_size=self.args["hidden_size"],
                                batch_first=True,
                                num_layers=1,
                                bidirectional=True,
                                dropout=self.args["dropout"])
        # 10.pointer network (for task2)
        self.pointer_weight_a1 = Linear(self.args["hidden_size"]*4, 1)
        self.pointer_weight_a2 = Linear(self.args["hidden_size"]*4, 1)

    def forward(self, batch, train=True):
        def fuse_layer(a, b):
            """
            return pass a and b to a fuse layer, not change dim.
            :param a:
            :param b:
            :return:
            """
            mab = F.tanh(self.fuse_m_layer(torch.cat([a, b, a * b, a - b], dim=-1)))
            gab = F.sigmoid(self.fuse_g_layer(torch.cat([a, b, a * b, a - b], dim=-1)))
            return gab * mab + (1 - gab) * a
        # get question and paras
        q_word, q_len = batch.q_word[0], batch.q_word[1]   # [b, max_q_len], [b]
        paras_word, paras_num, paras_lens = batch.paras_word[0], batch.paras_word[1], batch.paras_word[2]  # (b, max_p_num, max_p_len), (b), (b, max_p_num)
        batch_size, max_p_num, max_p_len = paras_word.shape[0], paras_word.shape[1], paras_word.shape[2]
        paras_word = paras_word.reshape(batch_size*max_p_num, -1)  # [b*max_p_num, max_p_len]
        paras_lens = paras_lens.reshape(-1)  # [b*max_p_num]

        # build mask matrix
        paras_mask = seq_mask(paras_num, device=paras_num.device)  # (b, max_para_num)
        paras_word_mask = seq_mask(paras_lens, device=paras_lens.device).reshape(batch_size, max_p_num, -1)  # (b, max_para_num, max_p_len)

        # 1. word embedding layer
        q_word = self.word_emb(q_word)  # [b, max_q_len, d]
        paras_word = self.word_emb(paras_word)  # [b*max_para_num, max_p_len, d]

        # 2.contextual embedding layer
        q = self.context_lstm((q_word, q_len))[0]  # [b, max_q_len, 2*d]
        paras = self.context_lstm((paras_word, paras_lens))[0]  # [b*max_p_num, max_p_len,  2*d]

        # 3. co-attention layer
        # [b, max_q_len, 2*d] ---> [b*max_p_num, max_q_len, 2*d]
        q_tile = repeat_tensor(q, 0, max_p_num)
        #  [b*max_p_num, max_p_len, 1] * [b*max_p_num, max_q_len, 1] ----> [b*max_p_num, max_p_len, max_q_len]
        att_weight = torch.bmm(F.relu(self.att_weight_co(paras)), F.relu(self.att_weight_co(q_tile)).transpose(2, 1))
        att_weight = F.softmax(att_weight, dim=-1)
        # [b*max_p_num, max_p_len, max_q_len] * [b*max_p_num, max_q_len,  2*d]--> [b*max_p_num, max_p_len, 2*d]
        paras2q_co = torch.bmm(att_weight, q_tile)

        # 4. fuse layer1 to combine
        paras_combine = fuse_layer(paras, paras2q_co)  # [b*max_p_num, max_p_len,  2*d]

        # 5. self-attention layer
        # [b*max_p_num, max_p_len,  2*d] * [b*max_p_num, 2*d,  max_p_len] -> [b*max_p_num, max_p_len, max_p_len]
        self_att_weight = F.softmax(torch.bmm(paras_combine, paras_combine.transpose(2, 1)), dim=-1)
        # [b*max_p_num, max_p_len, max_p_len] * [b*max_p_num, max_p_len,  2*d] = [b*max_p_num, max_p_len,  2*d]
        paras_self_att = torch.bmm(self_att_weight, paras_combine)

        # 6. fuse layer2 to combine
        paras_combine = fuse_layer(paras_combine, paras_self_att)   # [b*max_p_num, max_p_len,  2*d]

        # 7. question self-align layer
        # [b, max_q_len, 2*d] ---> [b, max_q_len]
        align_weight_q = F.softmax(self.align_weight_q(q).squeeze(2), dim=-1)
        # [b, 1, max_q_len] * [b, max_q_len, 2*d] --> [b,1, 2*d ]  --> [b, 2*d]
        rq = torch.bmm(align_weight_q.unsqueeze(1), q).squeeze(1)

        # 8. paras self-align layer
        #  [b*max_p_num, max_p_len,  2*d]  --->  [b*max_p_num, max_p_len]
        align_weight_p = F.softmax(self.align_weight_p(paras_combine).squeeze(2), dim=-1)
        # [b*max_p_num, 1, max_p_len] * [b*max_p_num, max_p_len,  2*d] ---> [b*max_p_num, 1,  2*d] --> [b*max_p_num, 2*d]
        rps = torch.bmm(align_weight_p.unsqueeze(1), paras_combine).squeeze(1)

        # 9. relevance score(for task 1)
        # [b, 2*d]  ---> [b*max_p_num, 2*d]
        rq_tile = repeat_tensor(rq, 0, max_p_num)
        # ---> [b*max_p_num, 1]  --> [b, max_p_num]
        pr_score = self.score_weight_qp(rq_tile, rps).squeeze(-1).view(batch_size, max_p_num) * paras_mask

        # 9. shared LSTM layer (for task2)
        if train:
            # [b*max_p_num, max_p_len,  2*d]  ---> [b, max_p_num, max_p_len,  2*d]
            paras_combine = paras_combine.view(batch_size, max_p_num, max_p_len, -1)
            # combine all paras   --- > [b, max_p_num*max_p_len,  2*d]
            paras_combine = paras_combine.view(batch_size, max_p_num*max_p_len, -1)
            # shared LSTM   ---> [b, max_p_num*max_p_len,  2*d]
            # [b*max_p_num] -- > [b]
            paras_lens_sum = torch.sum(paras_lens.view(batch_size, -1), 1)
            # [b, max_p_num*max_p_len,  2*d]----> [b, max_p_num*max_p_len,  2*d]
            gp = self.shared_lstm((paras_combine, paras_lens_sum), total_length=paras_combine.shape[1])[0]
            # print("gp.size: ", gp.size())
            # 10. pointer network (for task2)
            # [b, max_p_num*max_p_len, 4*d]---> [b, total_paras_lens]
            p1 = self.pointer_weight_a1(torch.cat((paras_combine, gp), dim=-1)).squeeze(2)
            # [b, max_p_num*max_p_len,  2*d]  -----> [b, max_p_num*max_p_len,  2*d]
            gp2 = self.shared_lstm((gp, paras_lens_sum), total_length=paras_combine.shape[1])[0]
            p2 = self.pointer_weight_a2(torch.cat((paras_combine, gp2), dim=-1)).squeeze(2)
            # mask p1 and p2
            concat_paras_word_mask = paras_word_mask.reshape(batch_size, -1)  # (b, max_para_num*max_p_len)
            p1 = -INF*(1-concat_paras_word_mask) + p1
            p2 = -INF*(1-concat_paras_word_mask) + p2
            # return :
            # p1, p2:  [b, max_p_num*max_p_len], score: [b, max_p_num]
            return p1, p2, pr_score
        else:
            # for dev and test, every para have to predict one answer
            # paras_combine: [b*max_p_num, max_p_len,  2*d]
            # paras_lens: [b*max_p_num]
            # gp: ---> [b*max_p_num, max_p_len,  2*d]
            gp = self.shared_lstm((paras_combine, paras_lens), total_length=paras_combine.shape[1])[0]
            # -->[b*max_p_num, max_p_len, 4*d]---> [b*max_p_num, max_p_len]
            p1 = self.pointer_weight_a1(torch.cat((paras_combine, gp), dim=-1)).squeeze(2)
            # gp2: [b*max_p_num, max_p_len, 2*d]
            gp2 = self.shared_lstm((gp, paras_lens), total_length=paras_combine.shape[1])[0]
            # --->[b*max_p_num, max_p_len, 4*d]---> [b*max_p_num, max_p_len]
            p2 = self.pointer_weight_a2(torch.cat((paras_combine, gp2), dim=-1)).squeeze(2)

            # choose max score*pr_score para
            # get answer for every para
            b_multi_para_num, p_len = p1.size()
            ls = nn.LogSoftmax(dim=1)
            # (b*max_para_num, c_len, c_len)
            # 下三角形mask矩阵(保证i<=j)
            mask = (torch.ones(p_len, p_len) * float('-inf')).to(paras_num.device
                                                                 ).tril(-1).unsqueeze(0).expand(b_multi_para_num, -1,
                                                                                                -1)
            # masked (b*max_para_num, c_len, c_len)
            score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
            # s_idx: [b*max_para_num, c_len]
            score, s_idx = score.max(dim=1)
            # e_idx: [b*max_para_num], score is for (s_idx, e_idx).
            score, e_idx = score.max(dim=1)
            s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze(-1)

            # reshape
            s_idx_re = s_idx.reshape(batch_size, -1)
            e_idx_re = e_idx.reshape(batch_size, -1)
            score_re = score.reshape(batch_size, -1)

            # choose max  score*pr_score para
            # mask for answer idx score, calc score product, get best para idx ----> [b]
            _, best_para_idx = torch.max(score_re * pr_score, dim=-1)
            # use best idx to choose s_idx and e_idx, s_idx: [b], e_idx:[b]
            s_idx = torch.gather(s_idx_re, 1, best_para_idx.view(-1, 1)).squeeze(-1)
            e_idx = torch.gather(e_idx_re, 1, best_para_idx.view(-1, 1)).squeeze(-1)
            return s_idx, e_idx, best_para_idx








