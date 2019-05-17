#!/usr/bin/python
# coding:utf-8

"""R-net
@author: yyhaker
@contact: 572176750@qq.com
@file: rnet.py
@time: 2019/5/9 18:12
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.nn import GRU, Linear
from utils import MatchRNN, seq_mask, PartiallyTrainEmbedding

INF = 1e30  # 定义正无穷


class RNet(nn.Module):
    """R-net model"""
    def __init__(self, args, pretrained, trainable_weight_idx):
        super(RNet, self).__init__()
        self.args = args["arch"]["args"]

        # 1. Word Embedding Layer
        # self.word_emb = PartiallyTrainEmbedding(pretrained, trainable_weight_idx)
        self.word_emb = nn.Embedding.from_pretrained(pretrained, freeze=True)

        # 2. Question & Passage Embedding Layer
        self.context_rnn = GRU(input_size=self.args["word_dim"],
                               hidden_size=self.args["hidden_size"],
                               bidirectional=True,
                               batch_first=True,
                               dropout=self.args["dropout"])

        # 3.Question-Passage Matching layer
        self.match_rnn = MatchRNN(mode="GRU",
                                  hp_input_size=self.args["hidden_size"]*2,
                                  hq_input_size=self.args["hidden_size"]*2,
                                  hidden_size=self.args["hidden_size"],
                                  bidirectional=True,
                                  gated_attention=True,
                                  dropout_p=self.args["dropout"],
                                  enable_layer_norm=True)

        # 4. Passage self-Matching layer
        self.self_match_rnn = MatchRNN(mode="GRU",
                                       hp_input_size=self.args["hidden_size"]*2,
                                       hq_input_size=self.args["hidden_size"]*2,
                                       hidden_size=self.args["hidden_size"],
                                       bidirectional=True,
                                       gated_attention=True,
                                       dropout_p=self.args["dropout"],
                                       enable_layer_norm=True)

        # 5. Modeling layer
        self.modeling_rnn1 = GRU(input_size=self.args["hidden_size"] * 2,
                                 hidden_size=self.args["hidden_size"],
                                 bidirectional=True,
                                 batch_first=True,
                                 dropout=self.args["dropout"])

        # 6. Pointer Network layer
        self.p1_weight_g = Linear(self.args["hidden_size"] * 2, 1, dropout=self.args["dropout"])
        self.p1_weight_m = Linear(self.args["hidden_size"] * 2, 1, dropout=self.args["dropout"])
        self.p2_weight_g = Linear(self.args["hidden_size"] * 2, 1, dropout=self.args["dropout"])
        self.p2_weight_m = Linear(self.args["hidden_size"] * 2, 1, dropout=self.args["dropout"])

        self.output_rnn = GRU(input_size=self.args["hidden_size"] * 2,
                              hidden_size=self.args["hidden_size"],
                              bidirectional=True,
                              batch_first=True,
                              dropout=self.args["dropout"])

        self.dropout = nn.Dropout(p=self.args["dropout"])

        # para ranking (for task 2)
        self.score_weight_qp = nn.Bilinear(self.args["hidden_size"] * 2, self.args["hidden_size"] * 2, 1)
        # self-align layer for question
        self.align_weight_q = Linear(self.args["hidden_size"] * 2, 1)
        # self-align layer for paras
        self.align_weight_p = Linear(self.args["hidden_size"] * 2, 1)

    def forward(self, batch):

        def pointer_network(g, m, l):
            """
            :param g: (batch,  p_len, d * 2)
            :param m: (batch, p_len , d * 2)
            :param l: (batch)
            :return: p1: (batch, p_len), p2: (batch, p_len)
            """
            # p1: (batch, p_len)
            p1 = (self.p1_weight_g(g) + self.p1_weight_m(m)).squeeze()
            # m2: (batch, p_len, d * 2)
            m2 = self.output_rnn((m, l))[0]
            # p2: (batch, p_len)
            p2 = (self.p2_weight_g(g) + self.p2_weight_m(m2)).squeeze()
            return p1, p2

        # 0. read data
        # ---->(b, max_q_len), (b)
        q_word, q_lens = batch["q_word"], batch["q_lens"]
        # ----> (b, max_para_num, max_para_len), (b), (b, max_para_num)
        paras_word, paras_num, paras_lens = batch["paras_word"], batch["paras_num"], batch["paras_lens"]
        # b, max_para_num, max_para_len
        batch_size, max_para_num, max_para_len = paras_word.shape[0], paras_word.shape[1], paras_word.shape[2]

        # scale batch size: ----> (b*max_para_num, max_para_len)
        paras_word = paras_word.reshape(max_para_num * batch_size, -1)
        # duplicate q: (b, max_q_len) -> (b, max_para_num, max_q_len) -> (b*max_para_num, max_q_len)
        qs_word = torch.stack([q_word for i in range(max_para_num)], dim=1).reshape(-1, q_word.shape[1])
        # duplicate q_lens: (b) ---> (b, max_para_num) ----> (b*max_para_num)
        qs_lens = torch.stack([q_lens for i in range(max_para_num)], dim=1).reshape(-1)

        # mask paras and qs
        # ---> (b, max_para_num)
        paras_nums_mask = seq_mask(paras_num, max_len=max_para_num, device=paras_num.device)
        # --->(b*max_para_num)
        paras_lens = paras_lens.reshape(-1)
        # --->(b*max_para_num, max_para_len)
        paras_word_mask = seq_mask(paras_lens, max_len=max_para_len, device=paras_lens.device)
        # --->(b, max_q_len)
        q_word_mask = seq_mask(q_lens, max_len=q_word.shape[1], device=q_lens.device)
        # --->(b*max_para_num, max_q_len)
        qs_word_mask = torch.stack([q_word_mask for i in range(max_para_num)], dim=1).reshape(-1, q_word.shape[1])

        # 1. Word Embedding Layer
        paras_word = self.word_emb(paras_word)  # (b*max_para_num, max_para_len, d)
        qs_word = self.word_emb(qs_word)  # (b*max_para_num, max_q_len, d)

        # 2. Question & Passage Embedding Layer
        paras = self.context_rnn((paras_word, paras_lens))[0]  # (b*max_para_num, max_para_len, 2*d)
        qs = self.context_rnn((qs_word, qs_lens))[0]  # (b*max_para_num, max_q_len, 2*d)

        # 3. gated attention-based recurrent networks
        q_aware_paras, _, _ = self.match_rnn.forward(paras.permute([1, 0, 2]), paras_word_mask,
                                                     qs.permute([1, 0, 2]), qs_word_mask)  # (max_para_len, b*max_para_num, 2*d)

        # 5. Passage self-Matching layer
        self_match_paras, _, _ = self.self_match_rnn.forward(q_aware_paras, paras_word_mask,
                                                             q_aware_paras, paras_word_mask)  # (max_para_len, b*max_para_num, 2*d)

        # 6. Modeling Layer
        m = self.modeling_rnn1((self_match_paras.permute([1, 0, 2]),
                                paras_lens))[0]  # (b*max_para_num, max_para_len, 2*d)

        # concat all paras
        concat_g = self_match_paras.permute([1, 0, 2]).reshape(batch_size, max_para_num*max_para_len, -1)
        concat_m = m.reshape(batch_size, max_para_num*max_para_len, -1)
        last_para_len = paras_lens.reshape(batch_size, -1)[:, -1]
        concat_para_lens = max_para_len * 2 + last_para_len

        # 7. pointer network layer
        p1, p2 = pointer_network(concat_g, concat_m, concat_para_lens)  # (b, paras_len), (b, paras_len)

        # mask res
        concat_paras_mask = paras_word_mask.reshape(batch_size, -1)  # (b, max_para_num*max_para_len)
        p1 = -INF * (1 - concat_paras_mask) + p1
        p2 = -INF * (1 - concat_paras_mask) + p2

        # (task2: )for paras ranking

        # self-align qs: (b*max_para_num, max_q_len, 2*d) ---> (b*max_para_num, max_q_len)
        align_weight_q = F.softmax(self.align_weight_q(qs).squeeze(2), dim=-1)
        # [b*max_para_num, 1, max_q_len] * [b*max_para_num, max_q_len, 2*d]
        #           --> [b*max_para_num,1, 2*d ]  --> [b*max_para_num, 2*d]
        rqs = torch.bmm(align_weight_q.unsqueeze(1), qs).squeeze(1)

        # self-align paras: m: (b*max_para_num, max_p_len, 2*d)  -->  (b*max_para_num, max_p_len)
        align_weight_p = F.softmax(self.align_weight_p(m).squeeze(2), dim=-1)
        # [b*max_para_num, 1, max_p_len] * [b*max_para_num, max_p_len,  2*d]
        #                 ---> [b*max_para_num, 1,  2*d] --> [b*max_para_num, 2*d]
        rps = torch.bmm(align_weight_p.unsqueeze(1), m).squeeze(1)

        # match score [b*max_para_num, 1]  --> [b, max_para_num]
        pr_score = self.score_weight_qp(rqs, rps).squeeze(-1).view(batch_size, max_para_num)
        pr_score = -INF * (1 - paras_nums_mask) + pr_score

        return p1, p2, pr_score
