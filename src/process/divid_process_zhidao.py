#!/usr/bin/python
# coding:utf-8
"""使用多个cpu处理数据
@author: Mingxiang Tuo
@contact: tuomx@qq.com
@file: divid_preprocessed.py
@time: 2019/4/3 19:10

# 处理Search文件，输入官方预处理文件，输出改进的预处理文件
"""
import os
import sys
# add current path to sys path
sys.path.append(os.getcwd())
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
        "paragraphs": "",
        "s_idx": 12,
        "e_idx": 13,
        "fake_answer": "",
        "yesno_answers": "",
        "match_score": 0.8  # the F1 of fake_answer and true answer
    }  # 训练一个找answer span的模型 + 判断yes_no, 可测试时候怎么做？
    """
    # read process data
    # read process data.
    datas = []
    with open(path, 'r', encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if (idx + 1) % 100 == 0:
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
            # find para (for zhidao and search)
            if "zhidao" in path:
                best_paras = find_zhidao_paras(sample, train)
            elif "search" in path:
                best_paras = find_search_paras(sample, train)
            else:
                raise Exception("not supported data processing!")
            # skip len(best_paras)=0 samples
            if len(best_paras) == 0:
                continue
            data["paragraph"] = choose_one_para(best_paras, sample["segmented_question"], recall)  # 当前使用单para
            data["paragraphs"] = best_paras  # multiple paras
            if train:
                data["fake_answer"], data["s_idx"], data["e_idx"], data["match_score"] \
                    = find_fake_answer_from_multi_paras(sample, data["paragraphs"])
            datas.append(data)
    # write to processed data file
    ensure_dir(os.path.split(save_path)[0])
    print("processed done! write to file!")
    with codecs.open(save_path, "w", encoding="utf-8") as f_out:
        for line in datas:
            json.dump(line, f_out, ensure_ascii=False)
            print("", file=f_out)


if __name__ == '__main__':
    path = sys.argv[1]
    save_path = sys.argv[2]
    preprocessd(path, save_path)
