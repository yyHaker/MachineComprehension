#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: preprocess.py
@time: 2019/4/5 16:07
"""
from utils import metric_max_over_ground_truths, f1_score, recall, split_list


def find_fake_answer_from_multi_paras(sample, paragraphs):
    """
    find the best answer from multiple paras.
    :param sample:
    :param paragraphs:
    :return:
    """
    best_fake_answer = []
    best_s_idx = 0
    best_e_idx = 0
    best_score = 0.
    best_para = []  # not return
    for para in paragraphs:
        fake_answer, s_idx, e_idx, score = find_fake_answer_2(sample, para)
        if score > best_score:
            best_score = score
            best_fake_answer = fake_answer
            best_s_idx = s_idx
            best_e_idx = e_idx
            best_para = para
    # print("answer match score: ", best_score)
    # make sure answer span is in one of the para and match score is not zero
    # assert _check_ans_span(best_para, best_s_idx, best_e_idx, best_score) is True
    return best_fake_answer, best_s_idx, best_e_idx, best_score


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
            if len(doc["segmented_paragraphs"]) != 0:
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
            c_para = c_para + ["<sep>"] + para
        # 截取一定的长度(默认500)
        c_para = c_para[: 500] if len(c_para) > 500 else c_para
        best_paras.append(c_para)
    return best_paras


def choose_one_para(paras, question, metric_fn):
    """choose only one para from pre choosed paras.
    :param paras:
    :param question:
    :param metric_fn: f1_score, recall or blue4
    :return:
    """
    if len(paras) == 1:
        return paras[0]
    else:
        best_para = []
        max_score = 0.
        for para in paras:
            score = metric_max_over_ground_truths(metric_fn, para, question)
            if score > max_score:
                max_score = score
                best_para = para
        # return if not none
        if len(best_para) == 0:
            return paras[0]
        else:
            return best_para


def find_fake_answer_2(sample, paragraph):
    """find paragraph and fake answer for one sample. (not skip paras)
    --------
    答案只会在某个para中，不会跨para.
    :param sample:
    :param paragraph: the choosed para. (title + 4paras)
    :return:
    """
    best_para = paragraph
    paras_list = split_list(best_para, "<sep>")
    title = paras_list[0]
    paras = paras_list[1:]  # answer not in title
    # get answer tokens
    answer_tokens = set()
    for segmented_answer in sample["segmented_answers"]:
        answer_tokens = answer_tokens | set([token for token in segmented_answer])
    # choose answer span
    best_start_idx = len(title) + 1
    best_end_idx = best_start_idx
    best_score = 0.
    relative_pos = len(title) + 1
    for para in paras:
        res = _find_answer_span_from_one_para(para, answer_tokens, sample["segmented_answers"])
        if res:
            # just calc index and update best score
            s_idx, e_idx, score = res
            if score > best_score:
                best_start_idx = s_idx + relative_pos
                best_end_idx = e_idx + relative_pos
                best_score = score
        # must change relative position for next
        relative_pos = relative_pos + len(para) + 1
    return best_para[best_start_idx: best_end_idx+1], best_start_idx, best_end_idx, best_score


def _find_answer_span_from_one_para(para, answer_tokens, ref_answers):
    """find best answer span from one para.
    :param para: para tokens list.
    :param answer_tokens: answer tokens set.
    :param ref_answers: ref answers list.
    :return: start_idx and end_idx (both contain)or None if not found.
    """
    best_start_idx = -1
    best_end_idx = -1
    best_score = 0.
    for start_idx in range(len(para)):
        if para[start_idx] not in answer_tokens:
            continue  # speed the process
        for end_idx in range(start_idx, len(para)):
            span_string = para[start_idx: end_idx+1]
            score = metric_max_over_ground_truths(f1_score, span_string, ref_answers)
            if score > best_score:
                best_start_idx = start_idx
                best_end_idx = end_idx
                best_score = score
            if score == 1.0:
                break
    if best_start_idx == -1 or best_end_idx == -1 or best_score <= 0.:
        return None
    else:
        return best_start_idx, best_end_idx, best_score


def _check_ans_span(paragraph, best_start_idx, best_end_idx, best_score):
    """make sure answer span is in one of the para and match score is not zero
    :param paragraph:
    :param best_start_idx:
    :param best_end_idx:
    :param best_score:
    :return:
    """
    best_para = paragraph
    paras_list = split_list(best_para, "<sep>")
    title = paras_list[0]
    paras = paras_list[1:]  # answer not in title
    exists = False
    for para in paras:
        if "".join(paragraph[best_start_idx: best_end_idx+1]) in "".join(para):
            exists = True
            break
    return exists and best_score > 0


def find_fake_answer(sample, paragraph):
    """find paragraph and fake answer for one sample.
    :param sample:
    :param paragraph: the choosed para.
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
