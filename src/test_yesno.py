#!/usr/bin/python
# coding:utf-8

"""
@author: Mingxiang Tuo
@contact: tuomx@qq.com
@file: test_yesno.py
@time: 2019/5/6 15:48
预测yesno类型的answer，作为yesno分类模型的输入
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
from du_evaluation_metric import calc_score


def predict(args):
    """
    use the best model to test...
    :param args:
    :return:
    """
    # get logger
    logger = logging.getLogger('MC')
    if "zhidao" in args.path:
        logger.info("use zhidao models...")
    elif "search" in args.path:
        logger.info("use search model....")
    else:
        # raise Exception("Unknown  models!")
        pass
    # load best model and params
    state = torch.load(args.path)
    config = state["config"]   # test file path is in config.json
    config['data_loader']['args']['process_info'] = args.process_info
    config['data_loader']['args']['vocab_cache'] = args.vocab_cache
    config['data_loader']['args']['train_batch_size'] = args.batch_size
    config['data_loader']['args']['dev_batch_size'] = args.batch_size
    state_dict = state["state_dict"]

    logger.info('Run with config:')
    logger.info(json.dumps(config, indent=True))

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(config)

    # add config run params
    # config['arch']['args']['char_vocab_size'] = len(data_loader.CHAR.vocab)
    config['arch']['args']['word_vocab_size'] = len(data_loader.WORD.vocab)
    sep_idx, eop_idx = data_loader.vocab.stoi['<sep>'], data_loader.vocab.stoi['<eop>']
    logger.info(f'idx:{sep_idx},{eop_idx}')

    device = config["data_loader"]["args"]["device"]

    # build model architecture
    model = getattr(module_arch, config['arch']['type'])(config, data_loader.vocab_vectors, torch.tensor([sep_idx, eop_idx]))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info(f"begin predict examples on {args.file}...")
    preds = []
    with torch.no_grad():
        # data_loader.test_iter.device = device
        if args.file == 'train':
            data_iter = data_loader.train_iter
        elif args.file == 'dev':
            data_iter = data_loader.eval_iter
        else:
            data_iter = data_loader.test_iter

        for batch_idx, data in enumerate(data_iter):
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

            # for multiple para: (batch, max_para_num, max_p_len)
            concat_paras_words_idx = data.paras_word[0].reshape(data.paras_word[0].shape[0], -1)
            max_para_len = data.paras_word[0].shape[2]
            for i in range(batch_size):
                pred = {}
                # get question id, answer, question
                filter_idxs = [data_loader.vocab.stoi['<pad>'], data_loader.vocab.stoi['<sep>'], data_loader.vocab.stoi['<eop>']]

                q_id = data.id[i]
                answer = concat_paras_words_idx[i][s_idx[i]:e_idx[i] + 1]
                answer_words = [data_loader.PARAS.vocab.itos[idx] for idx in answer if idx not in filter_idxs]
                answer = ''.join(answer_words)
                segmented_answer = ' '.join(answer_words)

                question = data.q_word[0][i]
                question_words = [data_loader.PARAS.vocab.itos[idx] for idx in question if idx not in filter_idxs]
                question = ''.join(question_words)
                segmented_question = ' '.join(question_words)

                # for para idx, s_idx: [batch]
                answer_para_idx = int(s_idx[i].item() // max_para_len)
                # for pred
                pred["question_id"] = q_id
                pred["segmented_question"] = segmented_question
                pred["question"] = question
                pred["segmented_answers"] = [segmented_answer]
                pred["answers"] = [answer]
                pred["question_type"] = data.question_type[i]
                pred["yesno_answers"] = []# not predict now
                pred["yesno_label"] = data.yesno_answers[i]
                preds.append(pred)
            if batch_idx % 10 == 0:
                logger.info("predict {} samples done!".format((batch_idx + 1) * batch_size))

    logger.info("write result to file....")
    predict_file = args.target
    ensure_dir(os.path.split(predict_file)[0])
    with codecs.open(predict_file, 'w', encoding='utf-8') as f:
        for pred in preds:
            json.dump(pred, f, ensure_ascii=False)
            print("", file=f)

    if args.eval:
        results = calc_score(predict_file, args.ref_file)
        logger.info("ROUGE-L :{}, BLUE-4: {}".format(results["ROUGE-L"], results["BLUE-4"]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MC')
    parser.add_argument('-b', '--batch_size', default=64, type=int, help='batch_size')
    parser.add_argument('-p', "--path", default="", type=str, help="best model directory")
    parser.add_argument('-i', '--process_info', default='', type=str, help="process_info in args")
    parser.add_argument('-v', '--vocab_cache', default='', type=str, help="vocab cache path")
    parser.add_argument('-t', "--target", default="./result/predict/result.json", type=str, help="predict result file")
    parser.add_argument('-r', "--ref_file", default="", type=str, help="ref file")
    parser.add_argument('-d', '--device', default='', type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('-e', '--eval', default=True, action='store_true', help='Whether evaluate the result')
    parser.add_argument('-f', '--file', default='test', help='run on train/dev/test')
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