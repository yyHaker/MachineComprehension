#!/usr/bin/python
# coding:utf-8

"""
@author: Mingxiang Tuo
@contact: tuomx@qq.com
@file: combine.py
@time: 2019/4/4 10:18
合并拆分与处理后的结果
"""
import os
import sys


if __name__ == '__main__':
    path = sys.argv[1]
    save_path = sys.argv[2]
    num = int(sys.argv[3])

    num = int(num)
    print("begining combine the result...")
    combine_dataset = []
    for i in range(num):
        with open(path+str(i), 'r', encoding='utf-8') as fin:
            print('read file', path+str(i))
            combine_dataset += fin.readlines()
    print(f'data size {len(combine_dataset)}')
    # make sure path exist
    if not os.path.exists(os.path.split(save_path)[0]):
        os.makedirs(os.path.split(save_path)[0])
    with open(save_path, 'w', encoding='utf-8') as fout:
        print(f'write file {save_path}')
        for d in combine_dataset:
            fout.write(d)
