#!/usr/bin/python
# coding:utf-8

"""
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
from utils.preprocess import find_search_paras


def preprocessd_multi_para(path, save_path, train=True):
    """ 多para预处理，暂时不计算answer span
    """
    # read process data.
    datas = []
    with open(path, 'r', encoding="utf-8") as f:
        for idx, line in enumerate(f):
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
            best_paras = find_search_paras(sample)
            if len(best_paras) == 0:
                    continue
            data["paragraphs"] = best_paras
            # # find answer span
            if train:
                # find one answer
                # data["fake_answer"], data["s_idx"], data["e_idx"], data["match_score"], data["answer_para_idx"], data["answer_idx"]\
                #     = find_fake_answer_from_multi_paras(data["paragraphs"], sample["segmented_answers"])
                # find multiple answers
                data["fake_answers"], data["s_idxs"], data["e_idxs"], data["match_scores"], data["answer_para_idxs"], data[
                    "answer_idxs"] = find_fake_answers_from_multi_paras(data["paragraphs"], sample["segmented_answers"])
            # post process to add tags
            data["paragraphs"] = post_process_paras_flags(best_paras, max_len=3)
            if not train or check_scores(data["match_scores"]):
                datas.append(data)
            if (idx + 1) % 1000 == 0:
                print("processed: {}".format(idx + 1))
                print("data len: {}".format(len(datas)))
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
    # train = sys.argv[3]
    # if train == '0':
    #     train = False
    # else:
    #     train = True
    # print(train)
    preprocessd_multi_para(path, save_path, train=True)
