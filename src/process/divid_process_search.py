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
from utils.preprocess import find_search_multi_paras


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
            data["paragraphs"] = find_search_multi_paras(sample)

            # # find answer span
            if train:
                data["fake_answer"], data["s_idx"], data["e_idx"], data["match_score"], data["answer_para_idx"] \
                    = find_fake_answer_from_multi_paras(sample, data["paragraphs"])
            if not train or data['match_score'] != 0:
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
