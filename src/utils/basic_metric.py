#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: basic_metric.py
@time: 2019/3/25 15:10
"""
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """
    This function calculates and returns the precision, recall and f1-score
    Args:
        metric_fn: metric function pointer which calculates scores according to corresponding logic.
        prediction: prediction string list to be matched.
        ground_truths: golden string list reference.
    Returns:
        floats of (p, r, f1)
    Raises:
        None
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def metric_max_over_ground_truths_with_idx(metric_fn, prediction, ground_truths):
    """
    This function calculates and returns the precision, recall and f1-score
    相比上个函数，增加了最佳匹配对应的ground_truths的idx
    Args:
        metric_fn: metric function pointer which calculates scores according to corresponding logic.
        prediction: prediction string list to be matched.
        ground_truths: golden string list reference.
    Returns:
        floats of (p, r, f1)
    Raises:
        None
    """
    max_score = -1
    max_idx = -1
    for idx, ground_truth in enumerate(ground_truths):
        score = metric_fn(prediction, ground_truth)
        if score > max_score:
            max_score = score
            max_idx = idx
    return max_score, max_idx


def precision_recall_f1(prediction, ground_truth):
    """
    This function calculates and returns the precision, recall and f1-score
    Args:
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of (p, r, f1)
    Raises:
        None
    """
    # 对于中文字符串，需要在每个字之间加空格
    if isinstance(prediction, str):
        prediction = " ".join(prediction)
    if isinstance(ground_truth, str):
        ground_truth = " ".join(ground_truth)

    if not isinstance(prediction, list):
        prediction_tokens = prediction.split()
    else:
        prediction_tokens = prediction
    if not isinstance(ground_truth, list):
        ground_truth_tokens = ground_truth.split()
    else:
        ground_truth_tokens = ground_truth
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    p = 1.0 * num_same / len(prediction_tokens)
    r = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * p * r) / (p + r)
    return p, r, f1


def recall(prediction, ground_truth):
    """
    This function calculates and returns the recall
    Args:
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of recall
    Raises:
        None
    """
    return precision_recall_f1(prediction, ground_truth)[1]


def f1_score(prediction, ground_truth):
    """
    This function calculates and returns the f1-score
    Args:
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of f1
    Raises:
        None
    """
    return precision_recall_f1(prediction, ground_truth)[2]


def blue4(prediction, ground_truth):
    """
        This function calculates and returns the blue4-score
        Args:
            prediction: prediction string or list to be matched
            ground_truth: golden string or list reference
        Returns:
            floats of f1
        Raises:
            None
    """
    prediction = prediction
    ground_truth = [ground_truth]
    # chencherry = SmoothingFunction()
    return sentence_bleu(ground_truth, prediction)


if __name__ == "__main__":
    pred= "我的和你的"
    gold = "我和你"
    value = precision_recall_f1(pred, gold)
    # print(value)
    print("calc blue4: ")
    groud_truth = ['孕妇', '能', '吃', '荔枝', '吗']
    prediction = ['荔枝', '含有', '丰富', '的', '营养', '元素', '。', '荔枝', '味', '甘', '、', '酸', '、', '性', '温', ',', '入', '心', '、', '脾', '、', '肝经', ',', '孕妇', '可以', '吃', '荔枝', '吗', '?', '答案', '是', '肯定', '的', '。', '孕妇', '可以', '适量', '的', '吃', '一些', '荔枝', '。', '但', '不宜', '多', '吃', ',', '孕妇', '吃', '荔枝', '每次', '以', '100', '—', '200', '克', '为', '宜', ',', '一般', '不要', '超过', '10', '颗', '。']
    score = blue4(groud_truth, prediction)
    print(score)
