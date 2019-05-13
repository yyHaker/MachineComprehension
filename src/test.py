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
    if "zhidao" in args.model:
        logger.info("use zhidao models...")
    elif "search" in args.model:
        logger.info("use search model....")
    else:
        # raise Exception("Unknown  models!")
        pass

    # load best model and params
    model_path = os.path.join(args.path, args.model)
    state = torch.load(model_path)
    config = state["config"]   # test file path is in config.json

    logger.info('Best result on dev is {}'.format(state['monitor_best']))
    config['data_loader']['args']['dev_batch_size'] = args.batch_size
    state_dict = state["state_dict"]

    # set test_file
    if not args.test_file:
        raise AssertionError('You should spacify the test file name (like search.test1.json)')
    else:
        config['data_loader']['args']['test_file'] = args.test_file

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
    logger.info("begin predict examples...")
    preds = []
    with torch.no_grad():
        # data_loader.test_iter.device = device
        data_iter = data_loader.eval_iter if args.on_dev else data_loader.test_iter
        for batch_idx, data in enumerate(data_iter):
            p1, p2, score = model(data)
            # p1, p2 = model(data)
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
                q_id = data.id[i]
                filter_idxs = [data_loader.vocab.stoi['<pad>'], data_loader.vocab.stoi['<sep>'], data_loader.vocab.stoi['<eop>']]
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
                pred["yesno_answers"] = []  # not predict now
                if args.on_dev:
                    # pred['s_idx'] = s_idx[i]
                    # pred['e_idx'] = e_idx[i]
                    pred['answer_para_idx'] = answer_para_idx
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MC')
    parser.add_argument('-b', '--batch_size', default=64, type=int, help='batch_size')
    parser.add_argument('-p', "--path", default="./result/dureader/saved", type=str, help="best model directory")
    parser.add_argument('-m', '--model', default=None, type=str, help="best model name(.pth)")
    parser.add_argument('-t', "--target", default="./result/predict/result.json", type=str,
                        help="prediction result file")
    parser.add_argument("--test_file", default="", type=str,
                        help="prediction result file")
    parser.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('--on_dev', default=False, action='store_true', help='Whether get pred_result on dev')
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
