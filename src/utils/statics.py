#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: statics.py
@time: 2019/4/5 10:44
"""
from collections import Counter


def statics(datas, rate=0.9, most_common=5, reversed=False):
    """statics some datas value.
    :param datas: a list of data.
    :param rate: above rate data distribution.
    :param most_common: most common rank.
    :param reversed: if false is in the accending order, else is in the decending order.
    :return:
    'mean: '
    'max: '
    'min: '
    ''
    """
    # total count
    print("total count: ", len(datas))
    # for mean, max, min, most common
    mean = sum(datas) / len(datas)
    max_value = max(datas)
    min_value = min(datas)
    raw_datas = Counter(datas)
    print("mean: {}, max: {}, min: {}".format(mean, max_value, min_value))
    print("most common {}: ".format(most_common))
    print(raw_datas.most_common(most_common))
    # 统计datas中小于等于v的所占占比例(rate)
    c_list = raw_datas.most_common()
    sort_c_list = sorted(c_list, key=lambda x: x[0], reverse=reversed)
    # for rate
    c_count = 0
    for v, c in sort_c_list:
        c_count += c
        if c_count / len(datas) >= rate:
            break
    if reversed is False:
        print("below {} is account for {}%".format(v, c_count / len(datas) * 100))
    else:
        print("above {} is account for {}%".format(v, c_count / len(datas) * 100))
