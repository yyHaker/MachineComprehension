#!/usr/bin/python
# coding:utf-8

"""
@author: Mingxiang Tuo
@contact: tuomx@qq.com
@file: divid_preprocessed.py
@time: 2019/4/3 19:10

# 处理Search文件，输入官方预处理文件，输出改进的预处理文件
"""

import math
import sys
import json
from utils import *


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


def _find_search_para(sample):
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
    best_para = []
    # 选择前三个document
    docs = []
    for doc in sample["documents"]:
        if doc["is_selected"]:
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
            if len(paras):
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


def _find_fake_answer(sample, paragraph):
    """find paragraph and fake answer for one sample.
    :param samlpe:
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
            span_string = best_para[start_idx: end_idx + 1]
            F1_score = metric_max_over_ground_truths(f1_score, span_string, sample["segmented_answers"])
            if F1_score > best_score:
                best_score = F1_score
                best_start_idx = start_idx
                best_end_idx = end_idx
                if F1_score == 1.0:
                    return best_para[best_start_idx: best_end_idx + 1], best_start_idx, best_end_idx, best_score
    return best_para[best_start_idx: best_end_idx + 1], best_start_idx, best_end_idx, best_score


def preprocessd(path, save_path, train=True):
    """preprocess the process data to a list of dict. (own preprocess method)
        1. 使用预先处理的已经分词的数据
        2. 使用一个sample的字段有：
            “question_id”: ,
            "question_type": ,
            "segmented_question": ,
            "documents": [
                                ["segmented_title":   ,  "segmented_paragraphs": []] ，
                                ["segmented_title":   ,  "segmented_paragraphs": []],
                                ["segmented_title":   ,  "segmented_paragraphs": []].
                       ]
            "segmented_answers": [ ] ,
         3. 仅仅使用前三篇的document, 使用每个document的所有title+paragraph替换paragraph（保证截取文本的长度不超过预先设置的最大长度(500)）
         4. 计算各个paragraph和问题的BLUE-4分数，以衡量paragraph和问题的相关性，在分数前K的paragraph中，选择最早出现的paragraph.
          (paragraph选好了)
        5. 对于每个答案，在paragraph中选择与答案F1分数最高的片段，作为这个答案的参考答案片段；如果只有一个答案的模型，
        选择任意一个答案或者F1分数最高的那个答案对应的最佳的片段作为参考答案片段，训练时使用。
    ----------
    # question_type: "YES_NO": 0, "DESCRIPTION": 1, "ENTITY": 2
    # cyesno_answers: "Yes": 0, "No": 1, "Depends": "2"
    :param path:
    :return:
    (文本均是分词后的结果)
     train_d = {
        "question_id": "",
        "question": "",
        "question_type": "",
        "paragraph": "",
        "s_idx": 12,
        "e_idx": 13,
        "fake_answer": "",
        "yesno_answers": "",
        "match_score": 0.8  # the F1 of fake_answer and true answer
    }  # 训练一个找answer span的模型 + 判断yes_no, 可测试时候怎么做？
    """
    # read process data.
    datas = []
    with open(path, 'r', encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if (idx + 1) % 10 == 0:
                print("processed: {}".format(idx + 1))
            sample = json.loads(line.strip())
            # just pass for no answer sample.(for train)
            if train:
                if "answers" not in sample.keys() or len(sample["answers"]) == 0:
                    continue
            # copy to data
            data = {}
            data["question_id"] = sample["question_id"]
            data["question"] = sample["segmented_question"]
            data["question_type"] = sample["question_type"]
            if "yesno_answers" in sample.keys():
                data["yesno_answers"] = sample["yesno_answers"]
            else:
                data["yesno_answers"] = []
            # find para
            data["paragraph"] = _find_search_para(sample)

            # find answer span
            if train:
                data["fake_answer"], data["s_idx"], data["e_idx"], data["match_score"] \
                    = _find_fake_answer(sample, data["paragraph"])
            datas.append(data)
    # write to processed data file
    print("processed done! write to file!")
    with codecs.open(save_path, "w", encoding="utf-8") as f_out:
        for line in datas:
            json.dump(line, f_out, ensure_ascii=False)
            print("", file=f_out)
    return


if __name__ == '__main__':
    path = sys.argv[1]
    save_path = sys.argv[2]
    preprocessd(path, save_path)
