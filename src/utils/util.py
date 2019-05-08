#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: util.py
@time: 2019/3/9 15:43
"""
import json
import os
import spacy
import codecs
import nltk
import jieba
import torch

spacy_en = spacy.load('en')


def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text.strip())]


def CN_tokenizer(text):
    return list(jieba.cut(text))


def dumpObj(obj, file):
    """
    dump object to file.
    :param obj:
    :param file:
    :return:
    """
    with open(file, 'w') as f:
        json.dump(obj, f)


def write_vocab(word2id, file):
    """
    write vocab to txt file.
    :param word2id:
    :param file: 'txt file'
    :return:
    """
    with codecs.open(file, 'w', encoding="utf-8") as f:
        for w in word2id.keys():
            f.write(w)
            f.write("\n")


def loadObj(file):
    """
    load object from file.
    :param file:
    :return:
    """
    with open(file, 'r') as f:
        obj = json.load(f)
    return obj


def ensure_dir(path):
    """
    ensure the dir exists.
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def split_list(alist, word):
    """
    split list into lists according to one word.
    :param alist:
    :param word:
    :return:
    ---------
    Example:
    >>>a = ["a", "sep", "b", "c", "<sep>", "hjlo", "Hi"]
    >>>split_list(a, "<sep>")
    >>>[ ["a"], ["b", "c"], ["hjlo", "hi"] ]
    >>>b = ["a", "sep", "b", "c", "<sep>", "hjlo", "Hi", "sep"]
    >>>split_list(b, "sep")
    >>>[ ["a"], ["b", "c"], ["<hjlo>", "hi"] ]
    """
    relative_pos = 0
    res = []
    for i in range(len(alist)):
        if alist[i] == word:
            res.append(alist[relative_pos: i])
            relative_pos = i + 1
        elif i == len(alist) - 1 and relative_pos != len(alist) - 1:
            res.append(alist[relative_pos:])
            break
    return res


def seq_mask(seq_len, device, max_len=None):
    '''
    :param seq_len:
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


def log_softmax_mask(A, mask, dim=1, epsilon=1e-12):
    '''
    applay log_softmax on A and consider mask
    :param A:
    :param mask:
    :param dim:
    :param epsilon:
    :return:
    '''
    # According to https://discuss.pytorch.org/t/apply-mask-softmax/14212/7
    A_max = torch.max(A, dim=dim, keepdim=True)[0]
    A_exp = torch.exp(A - A_max)
    A_exp = A_exp * mask  # this step masks
    A_log_softmax = torch.log(A_exp / (torch.sum(A_exp, dim=dim, keepdim=True) + epsilon))
    return A_log_softmax


def repeat_tensor(tensor, dim=0, times=2):
    """
    repeat tensor along a dim for times. (将某一行连续复制times次)
    :param tensor: [b, *]
    :param dim:
    :param times: repeat time.
    :return:  [b*times, *]
    -------
    Example:
    >>>a = torch.rand(3, 5)
    >>>a
    >>>tensor([[ 0.2531,  0.5096,  0.4381,  0.3080,  0.0861],
        [ 0.9698,  0.4726,  0.1682,  0.9173,  0.5383],
        [ 0.2139,  0.1363,  0.1656,  0.6204,  0.8971]])
    >>>repeat_tensor(a, 0, 2)
    >>>tensor([[ 0.2531,  0.5096,  0.4381,  0.3080,  0.0861],
        [ 0.2531,  0.5096,  0.4381,  0.3080,  0.0861],
        [ 0.9698,  0.4726,  0.1682,  0.9173,  0.5383],
        [ 0.9698,  0.4726,  0.1682,  0.9173,  0.5383],
        [ 0.2139,  0.1363,  0.1656,  0.6204,  0.8971],
        [ 0.2139,  0.1363,  0.1656,  0.6204,  0.8971]])
    """
    res = []
    for i in range(tensor.size(0)):
        ai = tensor[i].unsqueeze(0)
        if len(ai.size()) == 2:
            ai = ai.repeat(times, 1)  # [times, d]
        elif len(ai.size()) == 3:
            ai = ai.repeat(times, 1, 1)
        res.append(ai)
    return torch.cat(res, dim=dim)


if __name__ == "__main__":
    text = "I like playing computer games."
    sent = "I want to watch tv in living room"
    text2 = "网站赌博输钱报警有吗"
    print(CN_tokenizer(text2))
    print(tokenizer(sent))

    a = ["a", "<sep>", "b", "c", "<sep>", "hjlo", "Hi", "<sep>"]
    res = split_list(a, "<sep>")

    a = torch.rand(3, 5, 7)
    print("a: ")
    print(a)
    res = repeat_tensor(a, 0, 3)
    print("res: ")
    print(res)

