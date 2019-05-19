#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: post_process.py
@time: 2019/5/18 15:28
"""

import json

train_path = "data/dureader/process/2019/v2.0/search/three_para_multi_answers_filter/search.train.jsonl"
dev_path = "data/dureader/process/2019/v2.0/search/three_para_multi_answers_filter/search.dev.jsonl"

with open(dev_path, 'r', encoding="utf-8") as f:
    for idx, line in enumerate(f):
        sample = json.loads(line)
        print(sample["question_id"])
        