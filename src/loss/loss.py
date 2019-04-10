#!/usr/bin/python
# coding:utf-8

"""Common loss function.
@author: yyhaker
@contact: 572176750@qq.com
@file: loss.py
@time: 2019/3/9 15:36
"""
import torch


# def cross_entropy(output, target):
#     """
#     cross entropy loss.
#     :param output:
#     :param target:
#     :return:
#     """
#     loss = torch.nn.CrossEntropyLoss()
#     return loss(output, target)


def nll_loss(output, target):
    """
    cross entropy loss.
    :param output: the nagtive log prob of each class idx
    :param target: class idx
    :return:
    """
    loss = torch.nn.NLLLoss()
    return loss(output, target)