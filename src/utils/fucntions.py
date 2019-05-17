#!/usr/bin/python
# coding:utf-8

"""some useful functions for build models.
@author: yyhaker
@contact: 572176750@qq.com
@file: fucntions.py
@time: 2019/5/13 11:19
"""
import torch


def masked_flip(vin, mask, flip_dim=0):
    """
    flip a tensor
    :param vin: (..., batch, ...), batch should on dim=1, input batch with padding values
    :param mask: (batch, seq_len), show whether padding index
    :param flip_dim: dim to flip on
    :return:
    """
    length = mask.data.eq(1).long().sum(1)
    batch_size = vin.shape[1]

    flip_list = []
    for i in range(batch_size):
        cur_tensor = vin[:, i, :]
        cur_length = length[i]

        idx = list(range(cur_length - 1, -1, -1)) + list(range(cur_length, vin.shape[flip_dim]))
        idx = vin.new_tensor(idx, dtype=torch.long)

        cur_inv_tensor = cur_tensor.unsqueeze(1).index_select(flip_dim, idx).squeeze(1)
        flip_list.append(cur_inv_tensor)
    inv_tensor = torch.stack(flip_list, dim=1)
    return inv_tensor


def masked_softmax(x, m=None, dim=-1):
    """
    Softmax with mask.
    :param x: the Tensor to be softmaxed.
    :param m: mask.
    :param dim:
    :return:
    """
    if m is not None:
        m = m.float()
        x = x * m
    e_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
    if m is not None:
        e_x = e_x * m
    softmax = e_x / (torch.sum(e_x, dim=dim, keepdim=True) + 1e-6)
    return softmax


def seq_mask(seq_len, device, max_len=None):
    '''
    mask a seq.
    :param seq_len: [b]
    :param device:
    :param max_len:
    :return: mask matrix
    '''
    batch_size = seq_len.size(0)
    if not max_len:
        max_len = torch.max(seq_len)
    mask = torch.zeros((batch_size, max_len), device=device)
    for i in range(batch_size):
        for j in range(seq_len[i]):
            mask[i][j] = 1
    return mask


def convert2dict(data, train=True):
    """convert data  to dict
    :param data:
    :return:
    """
    res = {}
    res["id"] = data.id
    res["s_idx"] = data.s_idx
    res["e_idx"] = data.e_idx
    res["c_word"] = data.c_word
    res["c_char"] = data.c_char
    res["q_word"] = data.q_word
    res["q_char"] = data.q_char
    return res


if __name__ == "__main__":
    vin = torch.Tensor([
        [1, 2, 3, 2, 0, 0],
        [4, 2, 9, 67, 45, 0],
        [1, 3, 5, 67, 0, 0],
        [9, 2, 0, 0, 0, 0],
    ])
    vin = torch.randn(2, 6, 5)   # [b, seq_len, d]
    vin = vin .permute([1, 0, 2])
    print("vin size: ", vin.size())

    mask = torch.IntTensor([
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 0]
    ])
    t = masked_flip(vin, mask, flip_dim=0)
    print(t)

