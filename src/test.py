#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: test.py
@time: 2019/4/2 09:00
"""
import argparse
import json
import os
import logging

import torch

import data_loader.dureader as module_data
import model.bidaf as module_arch
from utils import ensure_dir
import codecs


def predict(args):
    """
    use the best model to test...
    :param args:
    :return:
    """
    # get logger
    logger = logging.getLogger('MC')
    # load best model and params
    model_path = os.path.join(args.path, args.model)
    state = torch.load(model_path)
    config = state["config"]   # test file path is in config.json
    state_dict = state["state_dict"]
    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(config)

    # add config run params
    config['arch']['args']['char_vocab_size'] = len(data_loader.CHAR.vocab)
    config['arch']['args']['word_vocab_size'] = len(data_loader.WORD.vocab)

    device = config["data_loader"]["args"]["device"]

    # build model architecture
    model = getattr(module_arch, config['arch']['type'])(config, data_loader.vocab_vectors)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info("begin predict examples...")
    preds = []
    with torch.no_grad():
        # data_loader.test_iter.device = device
        for batch_idx, data in enumerate(data_loader.test_iter):
            p1, p2 = model(data)
            # 统计得到的answers
            # (batch, c_len, c_len)
            batch_size, c_len = p1.size()
            ls = torch.nn.LogSoftmax(dim=1)
            mask = (torch.ones(c_len, c_len) * float('-inf')).to(device).tril(-1).unsqueeze(0).expand(batch_size,
                                                                                                           -1, -1)
            score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
            score, s_idx = score.max(dim=1)
            score, e_idx = score.max(dim=1)
            s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()

            for i in range(batch_size):
                pred = {}
                # get question id, answer, question
                q_id = data.id[i]
                answer = data.c_word[0][i][s_idx[i]:e_idx[i] + 1]
                answer = ' '.join([data_loader.WORD.vocab.itos[idx] for idx in answer])
                question = data.q_word[0][i]
                question = ' '.join([data_loader.WORD.vocab.itos[idx] for idx in question])
                # for pred
                pred["question_id"] = q_id
                pred["question"] = question
                pred["answers"] = [answer]
                pred["question_type"] = data.question_type[i]
                pred["yesno_answers"] = []  # not predict now
                preds.append(pred)
    logger.info("write result to file....")
    predict_file = args.target
    ensure_dir(os.path.split(predict_file)[0])
    with codecs.open(predict_file, 'w', encoding='utf-8') as f:
        for pred in preds:
            json.dump(pred, f)
            print("", file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MC')
    parser.add_argument('-p', "--path", default="./result/dureader/saved/BiDAF", type=str, help="best model directory")
    parser.add_argument('-m', '--model', default=None, type=str, help="best model name(.pth)")
    parser.add_argument('-t', "--target", default="./result/predict/result.json", type=str, help="prediction result file")
    parser.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # prpare logger
    logger = logging.getLogger('MC')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    predict(args)
