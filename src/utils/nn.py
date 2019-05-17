#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: nn.py
@time: 2019/3/22 20:28
"""
import torch
import torch.nn as nn


class LSTM(nn.Module):
    """LSTM"""
    def __init__(self, input_size, hidden_size, batch_first=False, num_layers=1, bidirectional=False, dropout=0.2):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           batch_first=batch_first)

        self.reset_params()
        self.dropout = nn.Dropout(p=dropout)

    def reset_params(self):
        for i in range(self.rnn.num_layers):
            nn.init.orthogonal_(getattr(self.rnn, f'weight_hh_l{i}'))
            nn.init.kaiming_normal_(getattr(self.rnn, f'weight_ih_l{i}'))
            nn.init.constant_(getattr(self.rnn, f'bias_hh_l{i}'), val=0)
            nn.init.constant_(getattr(self.rnn, f'bias_ih_l{i}'), val=0)
            getattr(self.rnn, f'bias_hh_l{i}').chunk(4)[1].fill_(1)

            if self.rnn.bidirectional:
                nn.init.orthogonal_(getattr(self.rnn, f'weight_hh_l{i}_reverse'))
                nn.init.kaiming_normal_(getattr(self.rnn, f'weight_ih_l{i}_reverse'))
                nn.init.constant_(getattr(self.rnn, f'bias_hh_l{i}_reverse'), val=0)
                nn.init.constant_(getattr(self.rnn, f'bias_ih_l{i}_reverse'), val=0)
                getattr(self.rnn, f'bias_hh_l{i}_reverse').chunk(4)[1].fill_(1)

    def forward(self, x):
        self.rnn.flatten_parameters()
        x, x_len = x
        x = x.size()[1]
        x = self.dropout(x)

        x_len_sorted, x_idx = torch.sort(x_len, descending=True)
        x_sorted = x.index_select(dim=0, index=x_idx)
        _, x_ori_idx = torch.sort(x_idx)

        x_packed = nn.utils.rnn.pack_padded_sequence(x_sorted, x_len_sorted, batch_first=True)
        x_packed, (h, c) = self.rnn(x_packed)

        # print("h size: ", h.size())  # [2, 36, 128]

        x = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True, total_length=total_length)[0]
        x = x.index_select(dim=0, index=x_ori_idx)
        h = h.permute(1, 0, 2).contiguous().view(-1, h.size(0) * h.size(2)).squeeze()
        h = h.index_select(dim=0, index=x_ori_idx)

        return x, h


class GRU(nn.Module):
    """GRU"""
    def __init__(self, input_size, hidden_size, batch_first=False, num_layers=1, bidirectional=True, dropout=0.2):
        super(GRU, self).__init__()

        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=bidirectional,
                          batch_first=batch_first)

        self.reset_params()
        self.dropout = nn.Dropout(p=dropout)

    def reset_params(self):
        for i in range(self.rnn.num_layers):
            nn.init.orthogonal_(getattr(self.rnn, f'weight_hh_l{i}'))
            nn.init.kaiming_normal_(getattr(self.rnn, f'weight_ih_l{i}'))
            nn.init.constant_(getattr(self.rnn, f'bias_hh_l{i}'), val=0)
            nn.init.constant_(getattr(self.rnn, f'bias_ih_l{i}'), val=0)
            getattr(self.rnn, f'bias_hh_l{i}').chunk(3)[1].fill_(1)

            if self.rnn.bidirectional:
                nn.init.orthogonal_(getattr(self.rnn, f'weight_hh_l{i}_reverse'))
                nn.init.kaiming_normal_(getattr(self.rnn, f'weight_ih_l{i}_reverse'))
                nn.init.constant_(getattr(self.rnn, f'bias_hh_l{i}_reverse'), val=0)
                nn.init.constant_(getattr(self.rnn, f'bias_ih_l{i}_reverse'), val=0)
                getattr(self.rnn, f'bias_hh_l{i}_reverse').chunk(3)[1].fill_(1)

    def forward(self, x):
        self.rnn.flatten_parameters()
        x, x_len = x
        total_length = x.size()[1]
        x = self.dropout(x)

        x_len_sorted, x_idx = torch.sort(x_len, descending=True)
        x_sorted = x.index_select(dim=0, index=x_idx)
        _, x_ori_idx = torch.sort(x_idx)

        x_packed = nn.utils.rnn.pack_padded_sequence(x_sorted, x_len_sorted, batch_first=True)
        x_packed, h = self.rnn(x_packed)

        x = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True, total_length=total_length)[0]
        x = x.index_select(dim=0, index=x_ori_idx)

        h = h.permute(1, 0, 2).contiguous().view(-1, h.size(0) * h.size(2)).squeeze()
        h = h.index_select(dim=0, index=x_ori_idx)

        return x, h


class Linear(nn.Module):
    """Linear"""
    def __init__(self, in_features, out_features, dropout=0.0):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        self.reset_params()

    def reset_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.linear(x)
        return x


class PartiallyTrainEmbedding(nn.Module):
    """Partially train embedding layer."""
    def __init__(self, weight, trainable_weight_idx):
        super().__init__()
        self.num_to_learn = trainable_weight_idx.shape[0]
        self.trainable_weight_idx = trainable_weight_idx
        self.trainable_weight = torch.nn.Parameter(torch.empty(self.num_to_learn, weight.shape[1]))
        torch.nn.init.kaiming_uniform_(self.trainable_weight)
        weight[trainable_weight_idx] = self.trainable_weight
        self.register_buffer('weight', weight)

    def forward(self, inp):
        self.weight.detach()
        self.weight[self.trainable_weight_idx] = self.trainable_weight
        return torch.nn.functional.embedding(
            inp, self.weight, None, None, 2.0, False, False)