#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: dureader_plus.py
@time: 2019/5/17 14:02
"""
import codecs
import json
from torchtext import data
import torchtext.vocab as vocab
from utils import *
import logging


class DuReaderPlus(object):
    """DuReader_Plus dataset loader"""
    def __init__(self, config):
        # logger
        self.logger = logging.getLogger('MC')
        # params
        self.config = config["data_loader"]["args"]
        # set path (for raw data)
        data_path = self.config["data_path"]

        # get data_path_l (for processed data (.jsonl and .pt))
        if "search" in self.config["train_file"]:
            data_path_process = os.path.join(data_path, "search")
        elif "zhidao" in self.config["train_file"]:
            data_path_process = os.path.join(data_path, "zhidao")
        else:
            raise Exception("not supported data set now!")
        data_path_process = os.path.join(data_path_process, self.config["process_info"])
        ensure_dir(data_path_process)
        # (for .pt)ls
        processed_dataset_path = data_path_process + "/torchtext/"
        train_examples_path = processed_dataset_path + f'{self.config["train_file"]}.pt'
        dev_examples_path = processed_dataset_path + f'{self.config["dev_file"]}.pt'
        test_examples_path = processed_dataset_path + f'{self.config["test_file"]}.pt'

        # define Field
        self.logger.info("construct data loader....")
        self.RAW = data.RawField()
        self.RAW.is_target = False     # 读取id值
        self.Q_WORD = data.Field(sequential=True, use_vocab=True, batch_first=True,
                                 tokenize=lambda x: x, lower=False, include_lengths=True)

        self.T_WORD = data.Field(sequential=True, use_vocab=True, batch_first=True,
                                 tokenize=lambda x: x, lower=False, include_lengths=False)
        # for multi para  [b, para_num, seq_len] or [b, para_num, seq_len, w_len]
        self.PARAS = data.NestedField(self.T_WORD, use_vocab=True, tokenize=lambda x: x, include_lengths=True)

        self.LABEL = data.Field(sequential=False, use_vocab=False, unk_token=None)
        self.ALL_LABELS = data.NestedField(self.LABEL, use_vocab=False, pad_token=0, dtype=torch.long)

        dict_fields = {'question_id': ('id', self.RAW),
                       'question': ('q_word', self.Q_WORD),
                       'question_type': ('question_type', self.RAW),
                       'yesno_answers': ('yesno_answers', self.RAW),
                       'paragraphs': ('paras_word', self.PARAS),
                       's_idxs': ('s_idxs', self.ALL_LABELS),
                       'e_idxs': ('e_idxs', self.ALL_LABELS),
                       'answer_para_idxs': ('answer_para_idxs', self.ALL_LABELS),
                       'match_scores': ('match_scores', self.RAW)
        }

        list_fields = [('id', self.RAW),
                       ('q_word', self.Q_WORD),
                       ('question_type', self.RAW),
                       ('yesno_answers', self.RAW),
                       ('paras_word', self.PARAS),
                       ('s_idxs', self.ALL_LABELS),
                       ('e_idxs', self.ALL_LABELS),
                       ('answer_para_idxs', self.ALL_LABELS),
                       ('match_scores', self.RAW)]

        test_dict_fields = {'question_id': ('id', self.RAW),
                            'question': ('q_word', self.Q_WORD),
                            'question_type': ('question_type', self.RAW),
                            'yesno_answers': ('yesno_answers', self.RAW),
                            'paragraphs': ('paras_word', self.PARAS),
                            }

        test_list_fields = [('id', self.RAW),
                            ('q_word', self.Q_WORD),
                            ('question_type', self.RAW),
                            ('yesno_answers', self.RAW),
                            ('paras_word', self.PARAS),
                            ]

        # judge if need to build dataSet
        if not os.path.exists(train_examples_path) or not os.path.exists(dev_examples_path):
            self.logger.info("build train dataSet....")
            self.train, self.dev = data.TabularDataset.splits(
                path=f'{data_path_process}',
                train=f'{self.config["train_file"]}l',
                validation=f'{self.config["dev_file"]}l',
                format='json',
                fields=dict_fields
            )
            # save preprocessed data
            ensure_dir(processed_dataset_path)
            torch.save(self.train.examples, train_examples_path)
            torch.save(self.dev.examples, dev_examples_path)
        else:
            self.logger.info("loading train dataSet.....")
            train_examples = torch.load(train_examples_path)
            dev_examples = torch.load(dev_examples_path)
            self.train = data.Dataset(examples=train_examples, fields=list_fields)
            self.dev = data.Dataset(examples=dev_examples, fields=list_fields)

        # for test data
        if not os.path.exists(test_examples_path):
            self.logger.info("build test dataSet....")
            self.test = data.TabularDataset(
                path=f'{data_path_process}/{self.config["test_file"]}l',
                format='json',
                fields=test_dict_fields
            )
            # save preprocessed data
            ensure_dir(processed_dataset_path)
            torch.save(self.test.examples, test_examples_path)
        else:
            self.logger.info("loading test dataSet......")
            test_examples = torch.load(test_examples_path)
            self.test = data.Dataset(examples=test_examples, fields=test_list_fields)

        # build vocab
        # vocab_cache_path = f"{data_path}/{self.config['vocab_cache']}"
        # if not os.path.exists(vocab_cache_path):
        self.logger.info("build vocab....")
        # self.CHAR.build_vocab(self.train, self.dev)
        self.PARAS.build_vocab(self.train.paras_word, self.train.q_word, self.dev.paras_word, self.dev.q_word)
        self.Q_WORD.vocab = self.PARAS.vocab

        # load pretrained embeddings
        Vectors = vocab.Vectors(self.config["pretrain_emd_file"])
        self.PARAS.vocab.load_vectors(Vectors)

        #     # save vocab cache
        #     self.logger.info("save vocab....")
        #     with open(vocab_cache_path, 'wb') as fout:
        #         pickle.dump(self.PARAS.vocab, fout)
        # else:
        #     # load vocab
        #     self.logger.info(f"load vocab from {vocab_cache_path} ....")
        #     with open(vocab_cache_path, 'rb') as fin:
        #         self.PARAS.vocab = pickle.load(fin)
        #         self.WORD.vocab = self.PARAS.vocab
        #         self.Q_WORD.vocab = self.PARAS.vocab

        # just for call easy
        self.vocab_vectors = self.PARAS.vocab.vectors
        self.vocab = self.PARAS.vocab

        # build iterators
        self.logger.info("building iterators....")

        self.train_iter = data.BucketIterator(dataset=self.train,
                                              batch_size=self.config["train_batch_size"],
                                              device=self.config["device"],
                                              shuffle=True)

        self.eval_iter = data.BucketIterator(dataset=self.dev,
                                             batch_size=self.config["dev_batch_size"],
                                             device=self.config["device"],
                                             sort_key=lambda x: max([max(para_len) for para_len in x.paras_word[2]]),
                                             sort_within_batch=False,
                                             shuffle=False)

        self.test_iter = data.BucketIterator(dataset=self.test,
                                             batch_size=self.config["dev_batch_size"],
                                             sort_key=lambda x: max([max(para_len) for para_len in x.paras_word[2]]),
                                             sort_within_batch=False,
                                             device=self.config["device"],
                                             shuffle=False)
