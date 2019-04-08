#!/usr/bin/python
# coding:utf-8

"""
@author: Mingxiang Tuo
@contact: tuomx@qq.com
@file: divid_preprocessed.py
@time: 2019/4/3 19:10

# 处理Search文件，输入官方预处理文件，输出改进的预处理文件
"""

from utils import *
from utils.preprocess import find_search_multi_paras, find_search_paras


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
                data["yesno_answers"] = sample["segmented_answers"]
            # find para
            data["paragraph"] = find_search_paras(sample)

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


def preprocessd_multi_para(path, save_path, train=True):
    """ 多para预处理，暂时不计算answer span
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
            data["preprocessed_docs"] = find_search_multi_paras(sample)
            for doc in data["preprocessed_docs"]:
                paras = doc.split('<sep>')[1:]

            # # find answer span
            # if train:
            #     data["fake_answer"], data["s_idx"], data["e_idx"], data["match_score"] \
            #         = _find_fake_answer(sample, data["paragraph"])
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
    preprocessd_multi_para(path, save_path)
