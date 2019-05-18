#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: train.py
@time: 2019/3/9 15:49
"""

import argparse
import json
import os
import logging

import torch
import data_loader.dureader_plus as module_data
import model as module_arch
import loss.loss as module_loss
import metric.metric as module_metric
from trainer import Trainer

import random
import numpy as np
from utils import ensure_dir
import codecs


def get_spacial_words(vocab):
    idxs = []
    words = ['<doc_0>', '<doc_1>', '<doc_2>', '<doc_3>', '<doc_4>',
             '<para_0>', '<para_1>', '<para_2>', '<para_3>', '<para_4>', '<para_5>', '<para_6>', '<para_7>', '<para_8>', '<para_9>',
             '<title>', '<empty>', '<eop>']
    for word in words:
        idx = vocab.stoi[word]
        # all unk word in vocab idx are
        if idx != vocab.stoi['<this_is_a_unk_word>']:
            idxs.append(idx)
        else:
            logger.info(f'unk functional:{word}')
    return idxs


def main(config, resume):
    """main project"""
    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(config)

    # for idx, data in enumerate(data_loader.eval_iter):
    #     print("idx: ", idx, " ", data)

    # add config run params
    # config['arch']['args']['char_vocab_size'] = len(data_loader.CHAR.vocab)
    config['arch']['args']['word_vocab_size'] = len(data_loader.vocab)
    spacial_words = get_spacial_words(data_loader.vocab)
    logger.info(f'idx:{spacial_words}')
    # build model architecture
    model = getattr(module_arch, config['arch']['type'])(config, data_loader.vocab_vectors, torch.tensor(spacial_words))

    # get function handles of loss
    loss = getattr(module_loss, config['loss']['type'])

    # get metrics
    metrics = [getattr(module_metric, config['metrics'])]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = getattr(torch.optim, config['optimizer']['type'])(trainable_params, **config['optimizer']['args'])
    # lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer)

    trainer = Trainer(model, loss, metrics, optimizer,
                      resume=resume,
                      config=config,
                      data_loader=data_loader)

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MC')
    parser.add_argument('-c', '--config', default="", type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-n', '--name', default=None, type=str,
                        help='the path name to save model')
    parser.add_argument('-s', '--seed', default=123, type=int,
                        help='Random seed')
    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    if args.config:
        # load config file
        config = json.load(open(args.config))
        if args.name:
            config['name'] = args.name
        # path = os.path.join(config['trainer']['save_dir'], config['name'])
    elif args.resume:
        # load config file from checkpoint, in case new config file is not given.
        # Use '--config' and '--resume' arguments together to load trained model and train more with changed config.
        config = torch.load(args.resume)['config']
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    # prpare logger
    logger = logging.getLogger('MC')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger.info('Run with config:')
    logger.info(json.dumps(config, indent=True))
    main(config, args.resume)
