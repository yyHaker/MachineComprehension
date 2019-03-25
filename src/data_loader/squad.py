#!/usr/bin/python
# coding:utf-8

"""squad 1.1 preprocess
@author: yyhaker
@contact: 572176750@qq.com
@file: squad.py
@time: 2019/3/22 20:35
"""
import json
import os
import nltk
import torch

from torchtext import data
import torchtext.vocab as vocab


def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]


class SQuAD(object):
    def __init__(self, config):
        # params
        args = config["data_loader"]["args"]
        # set path
        path = args["data_path"]
        dataset_path = path + '/torchtext/'
        train_examples_path = dataset_path + 'train_examples.pt'
        dev_examples_path = dataset_path + 'dev_examples.pt'

        print("preprocessing data files...")
        if not os.path.exists(f'{path}/{args["train_file"]}l'):
            self.preprocess_file(f'{path}/{args["train_file"]}')
        if not os.path.exists(f'{path}/{args["dev_file"]}l'):
            self.preprocess_file(f'{path}/{args["dev_file"]}')

        print("construct data loader...")
        self.RAW = data.RawField()      # 读取id值
        self.RAW.is_target = False
        self.CHAR_NESTING = data.Field(batch_first=True, tokenize=list, lower=True)
        self.CHAR = data.NestedField(self.CHAR_NESTING, tokenize=word_tokenize)
        self.WORD = data.Field(batch_first=True, tokenize=word_tokenize, lower=True, include_lengths=True)
        self.LABEL = data.Field(sequential=False, unk_token=None, use_vocab=False)

        dict_fields = {'id': ('id', self.RAW),
                       's_idx': ('s_idx', self.LABEL),
                       'e_idx': ('e_idx', self.LABEL),
                       'context': [('c_word', self.WORD), ('c_char', self.CHAR)],
                       'question': [('q_word', self.WORD), ('q_char', self.CHAR)]}

        list_fields = [('id', self.RAW), ('s_idx', self.LABEL), ('e_idx', self.LABEL),
                       ('c_word', self.WORD), ('c_char', self.CHAR),
                       ('q_word', self.WORD), ('q_char', self.CHAR)]

        if os.path.exists(dataset_path):
            print("loading splits...")
            train_examples = torch.load(train_examples_path)
            dev_examples = torch.load(dev_examples_path)

            self.train = data.Dataset(examples=train_examples, fields=list_fields)
            self.dev = data.Dataset(examples=dev_examples, fields=list_fields)
        else:
            print("building splits...")
            self.train, self.dev = data.TabularDataset.splits(
                path=path,
                train=f'{args["train_file"]}l',
                validation=f'{args["dev_file"]}l',
                format='json',
                fields=dict_fields)

            os.makedirs(dataset_path)
            torch.save(self.train.examples, train_examples_path)
            torch.save(self.dev.examples, dev_examples_path)

        # cut too long context in the training set for efficiency.
        if args["context_threshold"] > 0:
            self.train.examples = [e for e in self.train.examples if len(e.c_word) <= args["context_threshold"]]

        print("building vocab...")
        self.CHAR.build_vocab(self.train, self.dev)
        self.WORD.build_vocab(self.train, self.dev)

        # load pretrained vectors
        Vectors = vocab.Vectors(args["pretrain_emd_file"])
        self.WORD.vocab.load_vectors(Vectors)
        # just for call easy
        self.vocab_vectors = self.WORD.vocab.vectors

        print("building iterators...")
        self.train_iter, self.eval_iter = \
            data.BucketIterator.splits((self.train, self.dev),
                                       batch_sizes=[args["train_batch_size"], args["dev_batch_size"]],
                                       device=args["device"],
                                       sort_key=lambda x: len(x.c_word))

    def preprocess_file(self, path):
        """
        preprocess data to a list of dict.
        :param path:
        :return:
        """
        dump = []
        abnormals = [' ', '\n', '\u3000', '\u202f', '\u2009']

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data = data['data']

            for article in data:
                for paragraph in article['paragraphs']:
                    context = paragraph['context']
                    tokens = word_tokenize(context)
                    for qa in paragraph['qas']:
                        id = qa['id']
                        question = qa['question']
                        for ans in qa['answers']:
                            answer = ans['text']
                            s_idx = ans['answer_start']
                            e_idx = s_idx + len(answer)

                            l = 0
                            s_found = False
                            for i, t in enumerate(tokens):
                                while l < len(context):
                                    if context[l] in abnormals:
                                        l += 1
                                    else:
                                        break
                                # exceptional cases
                                if t[0] == '"' and context[l:l + 2] == '\'\'':
                                    t = '\'\'' + t[1:]
                                elif t == '"' and context[l:l + 2] == '\'\'':
                                    t = '\'\''

                                l += len(t)
                                if l > s_idx and s_found == False:
                                    s_idx = i
                                    s_found = True
                                if l >= e_idx:
                                    e_idx = i
                                    break

                            dump.append(dict([('id', id),
                                              ('context', context),
                                              ('question', question),
                                              ('answer', answer),
                                              ('s_idx', s_idx),
                                              ('e_idx', e_idx)]))
        # a list of dict
        with open(f'{path}l', 'w', encoding='utf-8') as f:
            for line in dump:
                json.dump(line, f)
                print('', file=f)


if __name__ == "__main__":
    config = json.load(open("../config.json"))
    print('loading SQuAD data...')
    # config["data_loader"]["args"]["data_path"] = '../data/squad'
    # config["data_loader"]["args"]["pretrain_emd_file"] = ''
    data = SQuAD(config)
    for i, data in enumerate(data.eval_iter):
        print(data)
