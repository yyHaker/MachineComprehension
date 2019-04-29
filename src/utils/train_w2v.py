#!/usr/bin/python
# coding:utf-8

"""
@author: Mingxiang Tuo
@contact: tuomx@qq.com
@file: train_w2v.py
@time: 2019/4/21 16:10
"""

import gensim, logging
import json
import argparse


class MySentences(object):
    def __init__(self, paths):
        self.paths = paths

    def __iter__(self):
        for path in self.paths:
            with open(path) as fin:
                for line in fin:
                    sample = json.loads(line.strip())
                    yield sample['segmented_question']
                    for doc in sample['documents']:
                        yield doc['segmented_title']
                        for paras in doc['segmented_paragraphs']:
                            yield paras


def parse_args():
    parser = argparse.ArgumentParser('Train Word2Vec.')
    parser.add_argument('--min_count', type=int, default=16)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--iter', type=int, default=10)
    parser.add_argument('--train_files', nargs='+', default=['./v2.0/search.train.json', './v2.0/search.dev.json', './v2.0/search.test1.json'],
                        help='list of files that contain the train data')
    parser.add_argument('--save_name', type=str, default='train_on_search.128.w2v')
    return parser.parse_args()


def train_word2vec(args):
    sentences = MySentences(args.train_files)
    model = gensim.models.Word2Vec(sentences, size=args.size, min_count=args.min_count, workers=args.workers, iter=args.iter)
    model.wv.save_word2vec_format(args.save_name)


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info('Run with args: {}'.format(args))

    train_word2vec(args)
