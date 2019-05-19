#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: post_process.py
@time: 2019/5/18 15:28
"""
from utils import *


def process(source_path, des_path):
    print("begin process ......")
    with open(source_path, 'r', encoding="utf-8") as f:
        results = []
        for idx, line in enumerate(f):
            sample = json.loads(line)
            sample["paragraphs"] = reverse_post_process_paras_flags(sample["paragraphs"], max_len=3)
            results.append(sample)

            if idx % 1000 == 0:
                print("processed {}".format(idx))
    print("processed done! write to file!")
    ensure_dir(os.path.split(des_path)[0])
    with open(des_path, "w", encoding="utf-8") as f_out:
        for line in results:
            json.dump(line, f_out, ensure_ascii=False)
            print("", file=f_out)


if __name__ == "__main__":
    search_train_path = "data/dureader/process/2019/v2.0/search/three_para_one_answer_filter/search.train.jsonl"
    search_dev_path = "data/dureader/process/2019/v2.0/search/three_para_one_answer_filter/search.dev.jsonl"
    search_test1_path = "data/dureader/process/2019/v2.0/search/three_para_one_answer_filter/search.test1.jsonl"

    to_search_train_path = "data/dureader/process/2019/v2.0/search/three_para_one_answer_filter_tag/search.train.jsonl"
    to_search_dev_path = "data/dureader/process/2019/v2.0/search/three_para_one_answer_filter_tag/search.dev.jsonl"
    to_search_test1_path = "data/dureader/process/2019/v2.0/search/three_para_one_answer_filter_tag/search.test1.jsonl"

    zhidao_train_path = "data/dureader/process/2019/v2.0/zhidao/three_para_one_answer_filter/zhidao.train.jsonl"
    zhidao_dev_path = "data/dureader/process/2019/v2.0/zhidao/three_para_one_answer_filter/zhidao.dev.jsonl"
    zhidao_test1_path = "data/dureader/process/2019/v2.0/zhidao/three_para_one_answer_filter/zhidao.test1.jsonl"

    to_zhidao_train_path = "data/dureader/process/2019/v2.0/zhidao/three_para_one_answer_filter_tag/zhidao.train.jsonl"
    to_zhidao_dev_path = "data/dureader/process/2019/v2.0/zhidao/three_para_one_answer_filter_tag/zhidao.dev.jsonl"
    to_zhidao_test1_path = "data/dureader/process/2019/v2.0/zhidao/three_para_one_answer_filter_tag/zhidao.test1.jsonl"

    # process(zhidao_train_path, to_zhidao_train_path)
    # process(zhidao_dev_path, to_zhidao_dev_path)
    # process(zhidao_test1_path, to_zhidao_test1_path)

    process(search_train_path, to_search_train_path)
    process(search_dev_path, to_search_dev_path)
    process(search_test1_path, to_search_test1_path)








