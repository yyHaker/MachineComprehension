#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: preprocess.py
@time: 2019/4/5 16:07
"""
from utils import *
from .filter import IllegalWords


def find_fake_answers_from_multi_paras(paragraphs, ref_answers):
    """
    find the best fake answers from multiple paras for every ref answer.
    :param paragraphs:
    :param ref_answers: ref answers list.
    :return:
    """
    best_fake_answers = []
    best_s_idxs = []
    best_e_idxs = []
    best_scores = []
    best_para_idxs = []
    best_ans_idxs = []
    # for every ref answer to choose the best fake answer
    for ref_ans in ref_answers:
        best_fake_answer, best_s_idx, best_e_idx, best_score, best_para_idx, best_ans_idx \
            = find_fake_answer_from_multi_paras(paragraphs, [ref_ans])
        # get result
        best_fake_answers.append(best_fake_answer)
        best_s_idxs.append(best_s_idx)
        best_e_idxs.append(best_e_idx)
        best_scores.append(best_score)
        best_para_idxs.append(best_para_idx)
        best_ans_idxs.append(best_ans_idx)
    return best_fake_answers, best_s_idxs, best_e_idxs, best_scores, best_para_idxs, best_ans_idxs


def find_fake_answer_from_multi_paras(paragraphs, ref_answers):
    """
    find the best answer from multiple paras.
    :param paragraphs:
    :param ref_answers: ref answers list.
    :return:
    """
    best_fake_answer = []
    best_s_idx = -1
    best_e_idx = -1
    best_score = 0.
    best_para_idx = -1
    best_ans_idx = -1
    for para_idx, para in enumerate(paragraphs):
        if para[0] == '<eop>':
            continue
        fake_answer, s_idx, e_idx, score, ans_idx = find_fake_answer_2(para, ref_answers)
        if score > best_score:
            best_score = score
            best_fake_answer = fake_answer
            best_s_idx = s_idx
            best_e_idx = e_idx
            best_para_idx = para_idx
            best_ans_idx = ans_idx
    # print("answer match score: ", best_score)
    # make sure answer span is in one of the para and match score is not zero
    # assert _check_ans_span(best_para, best_s_idx, best_e_idx, best_score) is True
    return best_fake_answer, best_s_idx, best_e_idx, best_score, best_para_idx, best_ans_idx


# def find_search_paras_2(sample):
#     """ find search paragraph
#     1. 选择is_select=True的前3篇doc，每个doc取前10个para
#     2. 将标题和各段中间插入连接符号<sep>拼接在一起，没有超过最大长度（500）则返回这个段落，否则执行3
#     3. 对于每个doc中的10个段落，计算各段落和问题的BLEU-4分数，来衡量段落和问题的相关性
#     4. 在排名前3的段落中找到idx最小的段落（越靠前越有用），然后将该段落和该段落的下一段落拼接（下一段落可能包含答案）
#     5. 在剩余8个段落中，每段选取第一句话
#     6.判断之前所有内容长度和是否超过500，超过则截取，保证所有doc中内容总共不超过500
#     :param sample:
#     :return: docs 格式如下：
#
#     [paragraph for doc1, paragraph for doc2, paragraph for doc3]
#
#     """
#     def first_sentence(para):
#         if not len(para):
#             return []
#         split_tag = ['。', '!', '?', '！', '？']
#         s = []
#         for word in para:
#             s.append(word)
#             if word in split_tag:
#                 break
#         if s[-1] not in split_tag:
#             s.append('。')
#         return s
#
#     concat_para = []
#     # 选择前三个document
#     docs = []
#     for doc in sample["documents"]:
#         if len(doc["segmented_paragraphs"]) != 0:
#             docs.append(doc)
#     docs = docs[:3]
#     # 先全部拼接
#     for doc in docs:
#         paras = doc["segmented_paragraphs"][:10]
#         concat_para += doc["segmented_title"]
#         concat_para += ['<sep>']
#         for para in paras:
#             concat_para += para + ['<sep>']
#     # 看全部拼接后长度是否超过阈值
#     if len(concat_para) <= 500:
#         preprocess_docs = []
#         for doc in docs:
#             preprocess_para = []
#             # 拼title
#             preprocess_para += doc["segmented_title"]
#             # 仅选取不为空的最多前10个段落
#             paras = []
#             for para in doc["segmented_paragraphs"]:
#                 if len(para) != 0:
#                     paras.append(para)
#             paras = paras[: 10]
#             # 拼接段落
#             for para in paras:
#                 preprocess_para += ['<sep>'] + para
#             preprocess_para = preprocess_para[:500]
#             preprocess_para += ['<eop>']
#             preprocess_docs.append(preprocess_para)
#     else:
#         preprocess_docs = []
#         question = sample['segmented_question']
#         all_len = 0
#         for doc in docs:
#             # 三篇总长度超过500则停止
#             if all_len >= 500:
#                 break
#             preprocess_para = []
#             # 拼title
#             preprocess_para += doc["segmented_title"] + ['<sep>']
#             # 仅选取不为空的前10个段落
#             paras = []
#             for para in doc["segmented_paragraphs"]:
#                 if len(para) != 0:
#                     paras.append(para)
#             paras = paras[: 10]
#
#             if paras:
#                 # 计算Recall
#                 prf_scores = [precision_recall_f1(para, question) for para in paras]
#                 scores = [i[1] for i in prf_scores]
#                 # 选取排名前2中最早出现的段落和下一段落
#                 scores_idx = [(i, scores[i]) for i in range(len(scores))]
#                 sorted_idx = sorted(scores_idx, key=lambda x: x[1], reverse=True)
#                 choose_idx = [i[0] for i in sorted_idx[:2]]
#                 # 拼接排名前2中最早出现的段落和下一段落
#                 early_idx = min(choose_idx)
#                 preprocess_para += paras[early_idx] + ['<sep>']
#                 early_next_idx = early_idx + 1
#                 if early_next_idx < len(paras):
#                     preprocess_para += paras[early_next_idx] + ['<sep>']
#                 # 拼剩余段落的第一句话
#                 for i in sorted_idx:
#                     if i != early_idx and i != early_next_idx:
#                         preprocess_para += first_sentence(paras[i[0]]) + ['<sep>']
# #                 preprocess_para = preprocess_para[:500]
#                 # 拼接前判断拼接后所有doc总长度是够超过500
#                 if len(preprocess_para) + all_len > 500:
#                     preprocess_para = preprocess_para[:500-all_len]
#                 preprocess_para += ['<eop>']
#                 preprocess_docs.append(preprocess_para)
#                 all_len += len(preprocess_para)
#     # 对para数量不足3的，直接加入<eop>
#     for i in range(3 - len(preprocess_docs)):
#         preprocess_docs.append(['<eop>'])
#     return preprocess_docs


def find_search_paras(sample):
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
    for idx, doc in enumerate(docs):
        paras = doc["segmented_paragraphs"][:10]
        concat_para += ["<doc_{}>".format(idx)] + ["<title>"]
        concat_para += doc["segmented_title"]
        concat_para += ['<sep>']
        for para in paras:
            concat_para += para + ['<sep>']
    # 看全部拼接后长度是否超过阈值
    if len(concat_para) <= 502:
        preprocess_docs = []
        for idx, doc in enumerate(docs):
            preprocess_para = []
            # 拼title
            preprocess_para += ["<doc_{}>".format(idx)] + ["<title>"]
            preprocess_para += doc["segmented_title"] + ['<sep>']
            # 仅选取不为空的最多前10个段落
            paras = []
            for para in doc["segmented_paragraphs"]:
                if len(para) != 0:
                    paras.append(para)
            paras = paras[: 10]
            # 拼接段落
            for para in paras:
                preprocess_para += para + ['<sep>']
            preprocess_para = preprocess_para[:502]
            preprocess_para += ['<eop>']
            preprocess_docs.append(preprocess_para)
    else:
        preprocess_docs = []
        question = sample['segmented_question']
        for idx, doc in enumerate(docs):
            preprocess_para = []
            # 拼title
            preprocess_para += ["<doc_{}>".format(idx)] + ["<title>"]
            preprocess_para += doc["segmented_title"] + ['<sep>']
            # 仅选取不为空的前10个段落
            paras = []
            for para in doc["segmented_paragraphs"]:
                if len(para) != 0:
                    paras.append(para)
            paras = paras[: 10]

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
                preprocess_para = preprocess_para[:502]
                preprocess_para += ['<eop>']
                preprocess_docs.append(preprocess_para)
    # 对para数量不足3的，直接加入<eop>
    # for i in range(3 - len(preprocess_docs)):
    #     preprocess_docs.append(['<eop>'])
    return preprocess_docs


def find_zhidao_paras(sample, train=True):
    """get zhidao paras.
    :param sample:
    :return:
      multiple paras,
       'best_paras:'  []
    ------
    every paragraph is composed:
    ['<doc_0>', '<title>', title, <sep>, para_0, <sep>, para_1, <sep>, para_2]
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
    for idx, doc in enumerate(docs):
        c_para = []
        title = doc["segmented_title"]
        # 取前面4个不为空的para
        paras = []
        for para in doc["segmented_paragraphs"]:
            if len(para) != 0:
                paras.append(para)
        # 只要没超过最大长度，就一直加，直到所有para加完为止
        # 将每个doc的tile+4paras拼接
        c_para = c_para + ["<doc_{}>".format(idx)] + ["<title>"] + title
        for para in paras:
            c_para = c_para + ["<sep>"] + para
            if len(c_para) > 502:
                break
        # 截取一定的长度(默认502)
        c_para = c_para[: 502] if len(c_para) > 502 else c_para
        best_paras.append(c_para)
    return best_paras


def find_raw_zhidao_paras(sample):
    """
    从官方预处理的数据中dureader 1.0找para。
    初步寻找前3个不为空的document的前4个不为空的paras，总共12个，不够的话使用
    :param sample:
    :return:
    """
    pass


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


def find_fake_answer_2(paragraph, ref_answers):
    """find paragraph and fake answer from one paragraph(combined by paras). (not skip paras)
    --------
    答案只会在某个para中，不会跨para.
    :param paragraph: the choosed para. (title + 4paras)
    :param ref_answers: ref answers list
    :return:
    """
    best_para = paragraph
    paras_list = split_list(best_para, "<sep>")
    title = paras_list[0]
    paras = paras_list[1:]  # answer not in title
    # get answer tokens
    answer_tokens = set()
    for segmented_answer in ref_answers:
        answer_tokens = answer_tokens | set([token for token in segmented_answer])
    # choose answer span
    best_start_idx = len(title) + 1
    best_end_idx = best_start_idx
    best_score = 0.
    best_ans_idx = -1
    relative_pos = len(title) + 1
    for para in paras:
        res = _find_answer_span_from_one_para(para, answer_tokens, ref_answers)
        if res:
            # just calc index and update best score
            s_idx, e_idx, score, ans_idx = res
            if score > best_score:
                best_start_idx = s_idx + relative_pos
                best_end_idx = e_idx + relative_pos
                best_score = score
                best_ans_idx = ans_idx
        # must change relative position for next
        relative_pos = relative_pos + len(para) + 1
    return best_para[best_start_idx: best_end_idx+1], best_start_idx, best_end_idx, best_score, best_ans_idx


def _find_answer_span_from_one_para(para, answer_tokens, ref_answers):
    """find best answer span from one para.
    :param para: para tokens list.
    :param answer_tokens: answer tokens set.
    :param ref_answers: ref answers list.
    :return: start_idx and end_idx (both contain)or None if not found.
    """
    best_ans_idx = -1
    best_start_idx = -1
    best_end_idx = -1
    best_score = 0.
    for start_idx in range(len(para)):
        if para[start_idx] not in answer_tokens:
            continue  # speed the process
        for end_idx in range(start_idx, len(para)):
            span_string = para[start_idx: end_idx+1]
            score, idx = metric_max_over_ground_truths_with_idx(f1_score, span_string, ref_answers)
            if score > best_score:
                best_ans_idx = idx
                best_start_idx = start_idx
                best_end_idx = end_idx
                best_score = score
            if score == 1.0:
                break
    if best_start_idx == -1 or best_end_idx == -1 or best_score <= 0.:
        return None
    else:
        return best_start_idx, best_end_idx, best_score, best_ans_idx


# def find_fake_answer(sample, paragraph):
#     """find paragraph and fake answer for one sample.
#     :param sample:
#     :param paragraph: the choosed para.
#     :return:
#     """
#     best_para = paragraph
#     # get answer tokens
#     answer_tokens = set()
#     for segmented_answer in sample["segmented_answers"]:
#         answer_tokens = answer_tokens | set([token for token in segmented_answer])
#     # choose answer span
#     best_start_idx = 0
#     best_end_idx = 0
#     best_score = 0.
#     for start_idx in range(len(best_para)):
#         if best_para[start_idx] not in answer_tokens:
#             continue  # speed the preprocess
#         for end_idx in range(start_idx, len(best_para)):
#             span_string = best_para[start_idx: end_idx+1]
#             F1_score = metric_max_over_ground_truths(f1_score, span_string, sample["segmented_answers"])
#             if F1_score > best_score:
#                 best_score = F1_score
#                 best_start_idx = start_idx
#                 best_end_idx = end_idx
#                 if F1_score == 1.0:
#                     return best_para[best_start_idx: best_end_idx+1], best_start_idx, best_end_idx, best_score
#     return best_para[best_start_idx: best_end_idx+1], best_start_idx, best_end_idx, best_score


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


def filter_illegal_words(sample):
    """
    filter illegal words.
    :param sample:
    :return:
    """
    illegal = IllegalWords()
    for idx, doc in enumerate(sample["documents"]):
        res_paras = []
        for para in doc["segmented_paragraphs"]:
            para = [w for w in para if w not in illegal.illegal_words]
            if len(para) != 0:
                res_paras.append(para)
        sample["documents"][idx]["segmented_paragraphs"] = res_paras
    return sample

