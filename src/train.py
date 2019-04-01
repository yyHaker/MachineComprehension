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

import torch
from utils import Logger

import data_loader.dureader as module_data
import model.bidaf as module_arch
import loss.loss as module_loss
import metric.metric as module_metric
from trainer import Trainer


# def get_instance(module, name, config, *args):
#     """get a instance of the object according to the params.
#     :param module: module
#     :param name: class name
#     :param config: config json file
#     :param args: position params
#     :return:
#     """
#     return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def main(config, resume):
    """main project"""
    # get logger
    train_logger = Logger()

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(config)

    # add config run params
    config['arch']['args']['char_vocab_size'] = len(data_loader.CHAR.vocab)
    config['arch']['args']['word_vocab_size'] = len(data_loader.WORD.vocab)

    # print some data
    for idx, data in enumerate(data_loader.eval_iter):
        print("idx: ", idx, " ", data)
        if idx > 5:
            break

    # build model architecture
    model = getattr(module_arch, config['arch']['type'])(config, data_loader.vocab_vectors)

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
                      data_loader=data_loader,
                      logger=train_logger)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MC')
    parser.add_argument('-c', '--config', default="du_config.json", type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    args = parser.parse_args()

    if args.config:
        # load config file
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
    elif args.resume:
        # load config file from checkpoint, in case new config file is not given.
        # Use '--config' and '--resume' arguments together to load trained model and train more with changed config.
        config = torch.load(args.resume)['config']
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    main(config, args.resume)
