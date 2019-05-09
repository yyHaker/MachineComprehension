#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: BiDAF.py
@time: 2019/3/22 20:19
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.nn import Linear, LSTM, PartiallyTrainEmbedding
from utils.util import log_softmax_mask, seq_mask

INF = 1e30  # 定义正无穷


class BiDAF(nn.Module):
    """BiDAF"""
    def __init__(self, args, pretrained):
        super(BiDAF, self).__init__()
        self.args = args["arch"]["args"]

        # 1. Character Embedding Layer
        # self.char_emb = nn.Embedding(self.args["char_vocab_size"], self.args["char_dim"], padding_idx=1)
        # nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)
        #
        # self.char_conv = nn.Conv2d(1, self.args["char_channel_size"],
        #                            (self.args["char_dim"], self.args["char_channel_width"]))

        # 2. Word Embedding Layer
        # initialize word embedding with GloVe
        self.word_emb = nn.Embedding.from_pretrained(pretrained, freeze=True)

        # highway network
        # assert self.args["hidden_size"] * 2 == (self.args["char_channel_size"] + self.args["word_dim"])
        self.seq_hidden = self.args["word_dim"]
        for i in range(2):
            setattr(self, f'highway_linear{i}',
                    nn.Sequential(Linear(self.seq_hidden, self.seq_hidden),
                                  nn.ReLU()))
            setattr(self, f'highway_gate{i}',
                    nn.Sequential(Linear(self.seq_hidden, self.seq_hidden),
                                  nn.Sigmoid()))

        # 3. Contextual Embedding Layer
        self.context_LSTM = LSTM(input_size=self.seq_hidden,
                                 hidden_size=self.args["hidden_size"],
                                 bidirectional=True,
                                 batch_first=True,
                                 dropout=self.args["dropout"])

        # 4. Attention Flow Layer
        self.att_weight_c = Linear(self.args["hidden_size"] * 2, 1)
        self.att_weight_q = Linear(self.args["hidden_size"] * 2, 1)
        self.att_weight_cq = Linear(self.args["hidden_size"] * 2, 1)

        # 5. Modeling Layer
        self.modeling_LSTM1 = LSTM(input_size=self.args["hidden_size"] * 8,
                                   hidden_size=self.args["hidden_size"],
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=self.args["dropout"])

        self.modeling_LSTM2 = LSTM(input_size=self.args["hidden_size"] * 2,
                                   hidden_size=self.args["hidden_size"],
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=self.args["dropout"])

        # 6. Output Layer
        self.p1_weight_g = Linear(self.args["hidden_size"] * 8, 1, dropout=self.args["dropout"])
        self.p1_weight_m = Linear(self.args["hidden_size"] * 2, 1, dropout=self.args["dropout"])
        self.p2_weight_g = Linear(self.args["hidden_size"] * 8, 1, dropout=self.args["dropout"])
        self.p2_weight_m = Linear(self.args["hidden_size"] * 2, 1, dropout=self.args["dropout"])

        self.output_LSTM = LSTM(input_size=self.args["hidden_size"] * 2,
                                hidden_size=self.args["hidden_size"],
                                bidirectional=True,
                                batch_first=True,
                                dropout=self.args["dropout"])

        self.dropout = nn.Dropout(p=self.args["dropout"])

    def forward(self, batch):
        # TODO: More memory-efficient architecture
        def char_emb_layer(x):
            """
            :param x: (batch, seq_len, word_len)
            :return: (batch, seq_len, char_channel_size)
            """
            batch_size = x.size(0)
            # (batch, seq_len, word_len, char_dim)
            x = self.dropout(self.char_emb(x))
            # (batch * seq_len, 1, char_dim, word_len)
            x = x.view(-1, self.args["char_dim"], x.size(2)).unsqueeze(1)
            # (batch * seq_len, char_channel_size, 1, conv_len) -> (batch * seq_len, char_channel_size, conv_len)
            x = self.char_conv(x).squeeze(2)
            # (batch * seq_len, char_channel_size, 1) -> (batch * seq_len, char_channel_size)
            x = F.max_pool1d(x, x.size(2)).squeeze()
            # (batch, seq_len, char_channel_size)
            x = x.view(batch_size, -1, self.args["char_channel_size"])

            return x

        def highway_network(x):
            """
            :param x1: (batch, seq_len, char_channel_size)
            :param x2: (batch, seq_len, word_dim)
            :return: (batch, seq_len, hidden_size * 2)
            """
            # (batch, seq_len, char_channel_size + word_dim)
            # x = torch.cat([x1, x2], dim=-1)
            for i in range(2):
                h = getattr(self, f'highway_linear{i}')(x)
                g = getattr(self, f'highway_gate{i}')(x)
                x = g * h + (1 - g) * x
            # (batch, seq_len, hidden_size * 2)
            return x

        def att_flow_layer(c, q):
            """
            :param c: (batch, c_len, hidden_size * 2)
            :param q: (batch, q_len, hidden_size * 2)
            :return: (batch, c_len, q_len)
            """
            c_len = c.size(1)
            q_len = q.size(1)

            # (batch, c_len, q_len, hidden_size * 2)
            #c_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1)
            # (batch, c_len, q_len, hidden_size * 2)
            #q_tiled = q.unsqueeze(1).expand(-1, c_len, -1, -1)
            # (batch, c_len, q_len, hidden_size * 2)
            #cq_tiled = c_tiled * q_tiled
            #cq_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1) * q.unsqueeze(1).expand(-1, c_len, -1, -1)

            cq = []
            for i in range(q_len):
                #(batch, 1, hidden_size * 2)
                qi = q.select(1, i).unsqueeze(1)
                #(batch, c_len, 1)
                ci = self.att_weight_cq(c * qi).squeeze()
                cq.append(ci)
            # (batch, c_len, q_len)
            cq = torch.stack(cq, dim=-1)

            # (batch, c_len, q_len)
            s = self.att_weight_c(c).expand(-1, -1, q_len) + \
                self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
                cq

            # (batch, c_len, q_len)
            a = F.softmax(s, dim=2)
            # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
            c2q_att = torch.bmm(a, q)
            # (batch, 1, c_len)
            b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
            # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
            q2c_att = torch.bmm(b, c).squeeze(1)
            # (batch, c_len, hidden_size * 2) (tiled)
            # print("q2c_attention size: ", q2c_att.size())  # ((batch, hidden_size * 2))
            q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
            # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

            # (batch, c_len, hidden_size * 8)
            x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
            return x

        def output_layer(g, m, l):
            """
            :param g: (batch, c_len, hidden_size * 8)
            :param m: (batch, c_len ,hidden_size * 2)
            :return: p1: (batch, c_len), p2: (batch, c_len)
            """
            # (batch, c_len)
            p1 = (self.p1_weight_g(g) + self.p1_weight_m(m)).squeeze(2)
            # (batch, c_len, hidden_size * 2)
            m2 = self.output_LSTM((m, l))[0]
            # (batch, c_len)
            p2 = (self.p2_weight_g(g) + self.p2_weight_m(m2)).squeeze(2)
            return p1, p2

        # choose para from the paras according to idx
        q_word, q_lens = batch.q_word[0], batch.q_word[1]  # (batch, q_len), (batch)
        p_word, p_num, p_lens = batch.paras_word[0], batch.paras_word[1], batch.paras_word[2]  # (batch, max_para_num, max_p_len), (batch), (batch, max_para_num)
        c_word = torch.cat([torch.index_select(p, 0, i).unsqueeze(0) for p, i in zip(p_word, batch.answer_para_idx)])  # (batch, 1, max_p_len)
        c_word = c_word.squeeze(1)   # (batch, max_p_len)
        c_lens = torch.cat([torch.index_select(length, 0, i) for length, i in zip(p_lens, batch.answer_para_idx)])  # (batch)
        # print("c_word shape: ", c_word.size())  # (batch, p_len)
        # 1. Character Embedding Layer
        # c_char = char_emb_layer(batch.c_char)
        # q_char = char_emb_layer(batch.q_char)
        # 2. Word Embedding Layer
        c_word = self.word_emb(c_word)
        q_word = self.word_emb(q_word)
        # print('word ebd layer:', c_word.shape, q_word.shape, c_lens.shape, q_lens.shape)
        # Highway network
        c = highway_network(c_word)
        q = highway_network(q_word)
        # print('Highway:', c.shape, q.shape)
        # 3. Contextual Embedding Layer
        c = self.context_LSTM((c, c_lens))[0]
        q = self.context_LSTM((q, q_lens))[0]
        # print('Contextual:', c.shape, q.shape)
        # 4. Attention Flow Layer
        g = att_flow_layer(c, q)
        # 5. Modeling Layer
        m = self.modeling_LSTM2((self.modeling_LSTM1((g, c_lens))[0], c_lens))[0]
        # 6. Output Layer
        p1, p2 = output_layer(g, m, c_lens)
        # (batch, c_len), (batch, c_len)
        return p1, p2


class BiDAFMultiParas(nn.Module):
    """BiDAF on multi paragraphs"""

    def __init__(self, args, pretrained, trainable_weight_idx):
        super(BiDAFMultiParas, self).__init__()
        self.args = args["arch"]["args"]

        # 1. Character Embedding Layer
        # self.char_emb = nn.Embedding(self.args["char_vocab_size"], self.args["char_dim"], padding_idx=1)
        # nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)
        #
        # self.char_conv = nn.Conv2d(1, self.args["char_channel_size"],
        #                            (self.args["char_dim"], self.args["char_channel_width"]))

        # 2. Word Embedding Layer
        # self.word_emb = nn.Embedding.from_pretrained(pretrained, freeze=True)
        self.word_emb = PartiallyTrainEmbedding(pretrained, trainable_weight_idx)

        # highway network
        # assert self.args["hidden_size"] * 2 == (self.args["char_channel_size"] + self.args["word_dim"])
        self.seq_hidden = self.args["word_dim"]
        for i in range(2):
            setattr(self, f'highway_linear{i}',
                    nn.Sequential(Linear(self.seq_hidden, self.seq_hidden),
                                  nn.ReLU()))
            setattr(self, f'highway_gate{i}',
                    nn.Sequential(Linear(self.seq_hidden, self.seq_hidden),
                                  nn.Sigmoid()))

        # 3. Contextual Embedding Layer
        self.context_LSTM = LSTM(input_size=self.seq_hidden,
                                 hidden_size=self.args["hidden_size"],
                                 bidirectional=True,
                                 batch_first=True,
                                 dropout=self.args["dropout"])

        # 4. Attention Flow Layer
        self.att_weight_p = Linear(self.args["hidden_size"] * 2, 1)
        self.att_weight_q = Linear(self.args["hidden_size"] * 2, 1)
        self.att_weight_pq = Linear(self.args["hidden_size"] * 2, 1)

        # 5. Modeling Layer
        self.modeling_LSTM1 = LSTM(input_size=self.args["hidden_size"] * 8,
                                   hidden_size=self.args["hidden_size"],
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=self.args["dropout"])

        self.modeling_LSTM2 = LSTM(input_size=self.args["hidden_size"] * 2,
                                   hidden_size=self.args["hidden_size"],
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=self.args["dropout"])

        # 6. Output Layer
        self.p1_weight_g = Linear(self.args["hidden_size"] * 8, 1, dropout=self.args["dropout"])
        self.p1_weight_m = Linear(self.args["hidden_size"] * 2, 1, dropout=self.args["dropout"])
        self.p2_weight_g = Linear(self.args["hidden_size"] * 8, 1, dropout=self.args["dropout"])
        self.p2_weight_m = Linear(self.args["hidden_size"] * 2, 1, dropout=self.args["dropout"])

        self.output_LSTM = LSTM(input_size=self.args["hidden_size"] * 2,
                                hidden_size=self.args["hidden_size"],
                                bidirectional=True,
                                batch_first=True,
                                dropout=self.args["dropout"])

        self.dropout = nn.Dropout(p=self.args["dropout"])

    def forward(self, batch):
        # TODO: More memory-efficient architecture
        def char_emb_layer(x):
            """
            :param x: (batch, seq_len, word_len)
            :return: (batch, seq_len, char_channel_size)
            """
            batch_size = x.size(0)
            # (batch, seq_len, word_len, char_dim)
            x = self.dropout(self.char_emb(x))
            # (batch * seq_len, 1, char_dim, word_len)
            x = x.view(-1, self.args["char_dim"], x.size(2)).unsqueeze(1)
            # (batch * seq_len, char_channel_size, 1, conv_len) -> (batch * seq_len, char_channel_size, conv_len)
            x = self.char_conv(x).squeeze(2)
            # (batch * seq_len, char_channel_size, 1) -> (batch * seq_len, char_channel_size)
            x = F.max_pool1d(x, x.size(2)).squeeze()
            # (batch, seq_len, char_channel_size)
            x = x.view(batch_size, -1, self.args["char_channel_size"])

            return x

        def highway_network(x):
            """
            :param x1: (batch, seq_len, char_channel_size)
            :param x2: (batch, seq_len, word_dim)
            :return: (batch, seq_len, hidden_size * 2)
            """
            # (batch, seq_len, char_channel_size + word_dim)
            # x = torch.cat([x1, x2], dim=-1)
            for i in range(2):
                h = getattr(self, f'highway_linear{i}')(x)
                g = getattr(self, f'highway_gate{i}')(x)
                x = g * h + (1 - g) * x
            # (batch, seq_len, hidden_size * 2)
            return x

        def att_flow_layer(p, q):
            """
            :param p: (batch, p_len, hidden_size * 2)
            :param q: (batch, q_len, hidden_size * 2)
            :return: (batch, p_len, q_len)
            """
            p_len = p.size(1)
            q_len = q.size(1)

            # (batch, p_len, q_len, hidden_size * 2)
            # p_tiled = p.unsqueeze(2).expand(-1, -1, q_len, -1)
            # (batch, p_len, q_len, hidden_size * 2)
            # q_tiled = q.unsqueeze(1).expand(-1, p_len, -1, -1)
            # (batch, p_len, q_len, hidden_size * 2)
            # pq_tiled = p_tiled * q_tiled
            # pq_tiled = p.unsqueeze(2).expand(-1, -1, q_len, -1) * q.unsqueeze(1).expand(-1, p_len, -1, -1)

            pq = []
            for i in range(q_len):
                # (batch, 1, hidden_size * 2)
                qi = q.select(1, i).unsqueeze(1)
                # (batch, p_len, 1)
                pi = self.att_weight_pq(p * qi).squeeze()
                pq.append(pi)
            # (batch, p_len, q_len)
            pq = torch.stack(pq, dim=-1)

            # (batch, p_len, q_len)
            s = self.att_weight_p(p).expand(-1, -1, q_len) + \
                self.att_weight_q(q).permute(0, 2, 1).expand(-1, p_len, -1) + \
                pq

            # (batch, p_len, q_len)
            a = F.softmax(s, dim=2)
            # (batch, p_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, p_len, hidden_size * 2)
            p2q_att = torch.bmm(a, q)
            # (batch, 1, p_len)
            b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
            # (batch, 1, p_len) * (batch, p_len, hidden_size * 2) -> (batch, hidden_size * 2)
            q2p_att = torch.bmm(b, p).squeeze()
            # (batch, p_len, hidden_size * 2) (tiled)
            q2p_att = q2p_att.unsqueeze(1).expand(-1, p_len, -1)
            # q2p_att = torch.stack([q2p_att] * p_len, dim=1)

            # (batch, p_len, hidden_size * 8)
            x = torch.cat([p, p2q_att, p * p2q_att, p * q2p_att], dim=-1)
            return x

        def output_layer(g, m, l):
            """
            :param g: (batch, para_num * p_len, hidden_size * 8)
            :param m: (batch, para_num * p_len ,hidden_size * 2)
            :param l: (batch)
            :return: p1: (batch, para_num * p_len), p2: (batch, para_num * p_len)
            """
            # p1: (batch, para_num*p_len)
            p1 = (self.p1_weight_g(g) + self.p1_weight_m(m)).squeeze()
            # m2: (batch, para_num*p_len, hidden_size * 2)
            m2 = self.output_LSTM((m, l), total_length=m.shape[1])[0]
            # print('In Pointer:', p1.shape, m2.shape)
            # p2: (batch, para_num*p_len)
            p2 = (self.p2_weight_g(g) + self.p2_weight_m(m2)).squeeze()
            return p1, p2

        # 1. Character Embedding Layer
        # p_char = char_emb_layer(batch.p_char)
        # q_char = char_emb_layer(batch.q_char)
        # 2. Word Embedding Layer
        q_word, q_lens = batch.q_word[0], batch.q_word[1]  # (batch, p_len), (batch)
        p_word, p_num, p_lens = batch.paras_word[0], batch.paras_word[1], batch.paras_word[2]  # (batch, para_num, p_len), (batch), (batch, para_num)
        batch_size = p_word.shape[0]
        max_para_num = p_word.shape[1]
        max_p_len = p_word.shape[2]
        # build mask matrix
        para_mask = seq_mask(p_num, device=p_num.device)  # (batch, para_num)
        p_lens_reshape = p_lens.reshape(-1)  # (batch*para_num)
        p_word_mask = seq_mask(p_lens_reshape, device=p_lens_reshape.device).reshape(p_lens.shape[0], p_lens.shape[1],
                                                                                     -1)  # (batch, para_num, p_len)
        # resghape p: (batch, para_num, p_len) -> (batch*para_num, p_len)
        # reshape para_mask: (batch, para_num) -> (batch*para_num, 1, 1)
        p_word = p_word.reshape(max_para_num * batch_size, -1)
        reshape_para_mask = para_mask.reshape(-1).unsqueeze(1).unsqueeze(1)
        # print('reshape size:', q_word.shape, q_lens.shape, p_word.shape, p_lens.shape, reshape_para_mask.shape)
        p_word = self.word_emb(p_word)
        q_word = self.word_emb(q_word)
        # print('word ebd layer:', p_word.shape, q_word.shape, p_lens.shape, q_lens.shape)
        # Highway network
        p = highway_network(p_word)
        q = highway_network(q_word)
        # print('Highway:', p.shape, q.shape)
        # 3. Contextual Embedding Layer
        p = self.context_LSTM((p, p_lens_reshape))[0]  # (batch*para_num, p_len, hidden)
        q = self.context_LSTM((q, q_lens))[0]  # (batch, q_len, hidden)
        # duplicate q: (batch, q_len) -> (batch*para_num, q_len)
        # duplicate q_lens: (batch) -> (batch*para_num)
        q = torch.stack([q for i in range(max_para_num)], dim=1).reshape(-1, q.shape[1], q.shape[2]) * reshape_para_mask
        #         q_lens = torch.stack([q_lens for i in range(3)], dim=0).reshape(-1) * para_mask * reshape_para_mask
        # print('Contextual:', p.shape, q.shape)
        # 4. Attention Flow Layer
        # TODO mask on attention
        g = att_flow_layer(p, q)
        # 5. Modeling Layer
        m = self.modeling_LSTM2((self.modeling_LSTM1((g, p_lens_reshape))[0], p_lens_reshape))[0]
        # print('modeling:', g.shape, m.shape)
        # 6. Output Layer
        # concat p: (batch*para_num, p_len, hidden) -> (batch, para_num*p_len, hidden)
        # g:(batch*para_num, p_len, q_len) -> (batch, para_num*p_len, q_len)
        # m:(batch*para_num, p_len, hidden) -> (batch, para_num*p_len, hidden)
        # origin_p_lens:(batch, para_num) -> (batch) last para len + 2 * max_para_num
        concat_g = g.reshape(batch_size, -1, g.shape[-1])
        concat_m = m.reshape(batch_size, -1, m.shape[-1])
        last_para_len = p_lens[:, -1]
        concat_p_lens = max_p_len * 2 + last_para_len
        # print('concat:', concat_g.shape, concat_m.shape, concat_p_lens.shape)
        # print('len before:', origin_p_lens)
        # print('len after:', concat_p_lens)
        p1, p2 = output_layer(concat_g, concat_m, concat_p_lens)
        # (batch, p_len), (batch, p_len)
        # print('pointer:', p1.shape, p2.shape)
        # mask，将padding位置-inf
        concat_p_word_mask = p_word_mask.reshape(p_word_mask.shape[0], -1)  # (batch, para_num*p_len)
        p1 = -INF*(1-concat_p_word_mask)+p1
        p2 = -INF*(1-concat_p_word_mask)+p2
        return p1, p2
