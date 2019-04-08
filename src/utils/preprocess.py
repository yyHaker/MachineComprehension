#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: preprocess.py
@time: 2019/4/5 16:07
"""
from utils import metric_max_over_ground_truths, f1_score, precision_recall_f1


def find_search_paras(sample, train=True):
    """ find search paragraph
    1. 选择is_select=True的前3篇doc，每个doc取前10个para
    2. 将标题和各段中间插入连接符号<sep>拼接在一起，没有超过最大长度（500）则返回这个段落，否则执行3
    3. 对于每个doc中的10个段落，计算各段落和问题的BLEU-4分数，来衡量段落和问题的相关性
    4. 在排名前3的段落中找到idx最小的段落（越靠前越有用），然后将该段落和该段落的下一段落拼接（下一段落可能包含答案）
    5. 在剩余8个段落中，每段选取第一句话
    6. 将上述所有内容拼接后，截取最大长度500返回
    :param sample:
    :return:
    """
    def first_sentence(para):
        if not len(para):
            return []
        split_tag = ['。', '!', '?', '！', '？']
        s = []
        for word in para:
            s.append(word)
            if word in split_tag:
                break
        if s[-1] not in split_tag:
            s.append('。')
        return s

    best_para = []
    # 选择前三个document
    docs = []
    for doc in sample["documents"]:
        if train:
            if doc["is_selected"]:
                docs.append(doc)
            else:
                pass
        else:
            docs.append(doc)
    docs = docs[:3]
    for doc in docs:
        paras = doc["segmented_paragraphs"][:10]
        best_para += doc["segmented_title"]
        best_para += ['<sep>']
        for para in paras:
            best_para += para
    # 看全部拼接后长度是否超过阈值
    if len(best_para) <= 500:
        return best_para
    else:
        best_para = []
        question = sample['segmented_question']
        for doc in docs:
            # 拼title
            best_para += doc["segmented_title"]
            best_para += ['<sep>']
            # 仅选取前10个段落
            paras = doc["segmented_paragraphs"][:10]
            if paras:
                # 计算Recall
                prf_scores = [precision_recall_f1(para, question) for para in paras]
                scores = [i[1] for i in prf_scores]
                # 选取排名前2中最早出现的段落和下一段落
                scores_idx = [(i, scores[i]) for i in range(len(scores))]
                sorted_idx = sorted(scores_idx, key=lambda x: x[1], reverse=True)
                choose_idx = [i[0] for i in sorted_idx[:2]]
                # 拼接排名前2中最早出现的段落和下一段落
                early_idx = min(choose_idx)
                best_para += paras[early_idx]
                early_next_idx = early_idx + 1
                if early_next_idx < len(paras):
                    best_para += paras[early_next_idx]
                # 拼剩余段落的第一句话
                for i in sorted_idx:
                    best_para += first_sentence(paras[i[0]])
                    if len(best_para) > 500:
                        break
                if len(best_para) > 500:
                    break
        # 截取最大长度500
        best_para = best_para[:500]
        return best_para


def find_search_multi_paras(sample):
    """ find search paragraph
    1. 选择is_select=True的前3篇doc，每个doc取前10个para
    2. 将标题和各段中间插入连接符号<sep>拼接在一起，没有超过最大长度（500）则返回这个段落，否则执行3
    3. 对于每个doc中的10个段落，计算各段落和问题的BLEU-4分数，来衡量段落和问题的相关性
    4. 在排名前3的段落中找到idx最小的段落（越靠前越有用），然后将该段落和该段落的下一段落拼接（下一段落可能包含答案）
    5. 在剩余8个段落中，每段选取第一句话
    6. 将上述所有内容拼接后，截取最大长度500返回
    :param sample:
    :return: docs 格式如下：

    [paragraph for doc1, paragraph for doc2, paragraph for doc3]

    """
    def first_sentence(para):
        if not len(para):
            return []
        split_tag = ['。', '!', '?', '！', '？']
        s = []
        for word in para:
            s.append(word)
            if word in split_tag:
                break
        if s[-1] not in split_tag:
            s.append('。')
        return s

    concat_para = []
    # 选择前三个document
    docs = []
    for doc in sample["documents"]:
        if len(doc["segmented_paragraphs"]) != 0:
            docs.append(doc)
    docs = docs[:3]
    # 先全部拼接
    for doc in docs:
        paras = doc["segmented_paragraphs"][:10]
        concat_para += doc["segmented_title"]
        concat_para += ['<sep>']
        for para in paras:
            concat_para += para + ['<sep>']
    # 看全部拼接后长度是否超过阈值
    if len(concat_para) <= 500:
        preprocess_docs = []
        for doc in docs:
            preprocess_para = []
            for para in doc["segmented_paragraphs"][:10]:
                preprocess_para += para + ['<sep>']
            preprocess_doc = {
                'title':doc['segmented_title'],
                'paragraph':preprocess_para
            }
            preprocess_docs.append(preprocess_doc)
        return preprocess_docs
    else:
        preprocess_docs = []
        question = sample['segmented_question']
        for doc in docs:
            preprocess_para = []
            # 拼title
            preprocess_para += doc["segmented_title"] + ['<sep>']
            # 仅选取前10个段落
            paras = doc["segmented_paragraphs"][:10]

            if paras:
                # 计算Recall
                prf_scores = [precision_recall_f1(para, question) for para in paras]
                scores = [i[1] for i in prf_scores]
                # 选取排名前2中最早出现的段落和下一段落
                scores_idx = [(i, scores[i]) for i in range(len(scores))]
                sorted_idx = sorted(scores_idx, key=lambda x: x[1], reverse=True)
                choose_idx = [i[0] for i in sorted_idx[:2]]
                # 拼接排名前2中最早出现的段落和下一段落
                early_idx = min(choose_idx)
                preprocess_para += paras[early_idx] + ['<sep>']
                early_next_idx = early_idx + 1
                if early_next_idx < len(paras):
                    preprocess_para += paras[early_next_idx] + ['<sep>']
                # 拼剩余段落的第一句话
                for i in sorted_idx:
                    if i != early_idx and i != early_next_idx:
                        preprocess_para += first_sentence(paras[i[0]]) + ['<sep>']
                preprocess_docs.append(preprocess_para)
        return preprocess_docs


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
