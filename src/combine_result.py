#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: combine_result.py
@time: 2019/4/4 09:00
"""
import argparse
import logging
import json


def combine(args):
    result_zhidao = args.zhidao_path
    result_search = args.search_path
    result = args.target_path
    preds = []
    # for zhidao
    with open(result_zhidao, 'r', encoding='utf-8') as zhidao_file:
        for i, d in enumerate(zhidao_file):
            preds.append(json.loads(d))
    # for search
    with open(result_search, 'r', encoding='utf-8') as search_file:
        for j, d in enumerate(search_file):
            preds.append(json.loads(d))
    # for no para search (test1)
    # with open('./result/predict/no_para_search.json', 'r', encoding='utf-8') as no_para_serach_file:
    #     for j, d in enumerate(no_para_serach_file):
    #         preds.append(json.loads(d))
    # combine all data
    with open(result, 'w', encoding='utf-8') as f:
        for d in preds:
            json.dump(d, f, ensure_ascii=False)
            print("", file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MC')
    parser.add_argument('-z', "--zhidao_path", default="./result/predict/zhidao_result.json", type=str, help="zhidao result path")
    parser.add_argument('-s', '--search_path', default="./result/predict/search_result.json", type=str, help="search result path")
    parser.add_argument('-t', '--target_path', default="./result/predict/result.json", type=str, help="target path")
    args = parser.parse_args()

    # prpare logger
    logger = logging.getLogger('MC')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    combine(args)
