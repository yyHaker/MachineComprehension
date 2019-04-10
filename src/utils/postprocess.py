#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: postprocess.py
@time: 2019/4/10 10:25
"""


def post_process_paras(paras, max_len=4):
    """
    add special token to empty paras.
    :param paras:
    :param max_len:
    :return:
    -------
    example:
    >>> a = [ ['me', 'hate', 'tes'],
                ['what', 'is'],
                ['my']]
    >>> post_process_paras(a, max_len=4)
            [ ['me', 'hate', 'tes', '<eop>'],
                ['what', 'is', '<eop>'],
                ['my', '<eop>'],
                ['<eop>']]
    """
    new_paras = []
    for para in paras:
        if len(para) != 0:
            para = para + ["<eop>"]
            new_paras.append(para)
    pad = ["<eop>"]
    c_len = len(new_paras)
    while c_len < max_len:
        new_paras.append(pad)
        c_len += 1
    return new_paras


if __name__ == "__main__":
    a = [['me', 'hate', 'tes'],
                ['what', 'is'],
                ['my']]
    print(post_process_paras(a, max_len=4))
