#!/usr/bin/python
# coding:utf-8

"""
@author: Mingxiang Tuo
@contact: tuomx@qq.com
@file: ensemble.py.py
@time: 2019/5/13 14:45
"""

import argparse
import json
import os
import logging

import torch
import torch.nn.functional as F

import data_loader.dureader as module_data
import model.bidaf as module_arch
from utils import ensure_dir
import codecs
from du_evaluation_metric import calc_score


def build_data(batch):
    input_data = {
        'q_word': batch.q_word[0],
        'q_lens': batch.q_word[1],
        'paras_word': batch.paras_word[0],
        'paras_num': batch.paras_word[1],
        'paras_lens': batch.paras_word[2],
    }
    return input_data


def ensemble(args):
    """
    use the best model to test...
    :param args:
    :return:
    """
    logger.info(f"Ensemble model paths: {args.paths}")

    # load all best models and their args
    models = []
    configs = []
    state_dicts = []
    for model_path in args.paths:
        state = torch.load(model_path)
        config = state["config"]  # test file path is in config.json
        logger.info('{} result on dev: {}'.format(config['name'], state['monitor_best']))
        logger.info('{} model args: '.format(config['name']))
        logger.info(json.dumps(config['arch'], indent=True))
        state_dict = state["state_dict"]
        state_dicts.append(state_dict)
        configs.append(config)

    # select a config as data config
    global_config = configs[0]
    global_config['data_loader']['args']['dev_batch_size'] = args.batch_size
    # set test_file
    if not args.test_file:
        raise AssertionError('You should spacify the test file name (like search.test1.json)')
    else:
        global_config['data_loader']['args']['test_file'] = args.test_file

    logger.info('Run with data config:')
    logger.info(json.dumps(global_config['data_loader']['args'], indent=True))

    # setup data_loader instances
    data_loader = getattr(module_data, global_config['data_loader']['type'])(global_config)

    # add config run params
    # config['arch']['args']['char_vocab_size'] = len(data_loader.CHAR.vocab)
    global_config['arch']['args']['word_vocab_size'] = len(data_loader.WORD.vocab)
    sep_idx, eop_idx = data_loader.vocab.stoi['<sep>'], data_loader.vocab.stoi['<eop>']
    logger.info(f'idx:{sep_idx},{eop_idx}')

    device = global_config["data_loader"]["args"]["device"]

    # load all models
    models = []
    for config, state_dict in zip(configs, state_dicts):
        logger.info(f'Loading model: {config["name"]}')
        # build model architecture
        model = getattr(module_arch, config['arch']['type'])(config, data_loader.vocab_vectors,
                                                             torch.tensor([sep_idx, eop_idx]))
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        models.append(model)

    logger.info("begin predict examples...")
    preds = []
    with torch.no_grad():
        # data_loader.test_iter.device = device
        data_iter = data_loader.eval_iter if args.on_dev else data_loader.test_iter
        for batch_idx, data in enumerate(data_iter):
            input_data = build_data(data)
            all_p1, all_p2 = [], []
            for model in models:
                p1, p2, _ = model(input_data)
                all_p1.append(F.softmax(p1, dim=1))
                all_p2.append(F.softmax(p2, dim=1))

            # 通过求均值得到ensemble后的p1，p2
            p1 = torch.stack(all_p1, dim=1).mean(dim=1)
            p2 = torch.stack(all_p2, dim=1).mean(dim=1)

            # (batch, c_len, c_len)
            batch_size, c_len = p1.size()
            mask = (torch.ones(c_len, c_len) * float('-inf')).to(device).tril(-1).unsqueeze(0).expand(batch_size,
                                                                                                      -1, -1)
            score = (torch.log(p1).unsqueeze(2) + torch.log(p2).unsqueeze(1)) + mask
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
                filter_idxs = [data_loader.vocab.stoi['<pad>'], data_loader.vocab.stoi['<sep>'],
                               data_loader.vocab.stoi['<eop>']]
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

    if args.on_dev:
        results = calc_score(predict_file, args.ref_file)
        logger.info("ROUGE-L :{}, BLUE-4: {}".format(results["ROUGE-L"], results["BLUE-4"]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MC')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        help='batch_size')
    parser.add_argument('-t', "--target", default="./result/predict/result.json", type=str,
                        help="prediction result file")
    parser.add_argument("--test_file", default="", type=str,
                        help="prediction result file")
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('--on_dev', default=False, action='store_true',
                        help='Whether get pred_result on dev')
    parser.add_argument('--paths', nargs='+',
                        help='Paths of all ensemble model')
    parser.add_argument('-r', "--ref_file", default="", type=str,
                        help="ref file")

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

    ensemble(args)