#!/usr/bin/python
# coding:utf-8

"""
@author: Mingxiang Tuo
@contact: tuomx@qq.com
@file: divide.py
@time: 2019/4/4 17:00
"""
import os
import math
import sys


def divid_data(data, num):
    """
    将数据拆分成num份
    :param data:
    :param num:
    :return:
    """
    all_data = [[] for i in range(num)]
    split_size = math.ceil(len(data) / num)
    for i in range(num):
        all_data[i] = data[i * split_size: (i + 1) * split_size]
    return all_data


if __name__ == '__main__':
    path = sys.argv[1]
    num = sys.argv[2]
    num = int(num)
    with open(path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    print("Data size:", len(data))

    all_data = divid_data(data, num)
    print('Split data size:{}\t,num:{}'.format(len(all_data[0]), len(all_data)))
    for idx, d in enumerate(all_data):
        path_dir, name = os.path.split(path)
        split_dir = os.path.join(path_dir, "split")
        split_path = os.path.join(split_dir, f'{name}__{idx}')
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
        print('Save data:', split_path)
        with open(split_path, 'w', encoding='utf-8') as f:
            for line in d:
                f.write(line)
