#!/usr/bin/python
# coding:utf-8

"""
@author: Mingxiang Tuo
@contact: tuomx@qq.com
@file: combine.py
@time: 2019/4/4 10:18
合并拆分与处理后的结果
"""

import sys
import json


if __name__ == '__main__':
    path = sys.argv[1]
    save_path = sys.argv[2]
    num = int(sys.argv[3])

    combine_dataset = []
    for i in range(num):
        with open(path+str(i)) as fin:
            print('read file', path+str(i))
            combine_dataset += fin.readlines()
    print(f'data size {len(combine_dataset)}')
    with open(save_path, 'w') as fout:
        print(f'write file {save_path}')
        for d in combine_dataset:
            fout.write(d)