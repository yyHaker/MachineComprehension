#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: preprocess.py
@time: 2019/4/5 16:07
"""
from utils import metric_max_over_ground_truths, f1_score


def find_search_paras(sample, train=True):
    """get search paras.
    :param sample:
    :param train:
    :return:
      multiple paras,
       'best_paras:'  []
    """
    pass


def find_zhidao_paras(sample, train=True):
    """get zhidao paras.
    :param sample:
    :return:
      multiple paras,
       'best_paras:'  []
    """
    # 计算最佳的paras
    best_paras = []
    # 取最前面三个的不为空的document
    docs = []
    if train:
        for doc in sample["documents"]:
            if doc["is_selected"] and len(doc["segmented_paragraphs"]) != 0:
                docs.append(doc)
    else:
        for doc in sample["documents"]:
            if len(doc["segmented_paragraphs"]) != 0:
                docs.append(doc)
    docs = docs[: 3]
    for doc in docs:
        c_para = []
        title = doc["segmented_title"]
        # 取前面4个不为空的para
        paras = []
        for para in doc["segmented_paragraphs"]:
            if len(para) != 0:
                paras.append(para)
        paras = paras[: 4]
        # 将每个doc的tile+4paras拼接
        c_para = c_para + title
        for para in paras:
            c_para = c_para + para
        # 截取一定的长度(默认500)
        c_para = c_para[: 500] if len(c_para) > 500 else c_para
        best_paras.append(c_para)
    return best_paras


def find_fake_answer(sample, paragraph):
    """find paragraph and fake answer for one sample.
    :param sample:
    :param paragraph:
    :return:
    """
    best_para = paragraph
    # get answer tokens
    answer_tokens = set()
    for segmented_answer in sample["segmented_answers"]:
        answer_tokens = answer_tokens | set([token for token in segmented_answer])
    # choose answer span
    best_start_idx = 0
    best_end_idx = 0
    best_score = 0.
    for start_idx in range(len(best_para)):
        if best_para[start_idx] not in answer_tokens:
            continue  # speed the preprocess
        for end_idx in range(start_idx, len(best_para)):
            span_string = best_para[start_idx: end_idx+1]
            F1_score = metric_max_over_ground_truths(f1_score, span_string, sample["segmented_answers"])
            if F1_score > best_score:
                best_score = F1_score
                best_start_idx = start_idx
                best_end_idx = end_idx
                if F1_score == 1.0:
                    return best_para[best_start_idx: best_end_idx+1], best_start_idx, best_end_idx, best_score
    return best_para[best_start_idx: best_end_idx+1], best_start_idx, best_end_idx, best_score
