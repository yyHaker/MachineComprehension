#!/usr/bin/python
# coding:utf-8

"""从dureader 2019.2.0的官方预处理数据的基础上采用启发式预处理的方式处理数据，加载数据
@author: yyhaker
@contact: 572176750@qq.com
@file: dureader.py
@time: 2019/3/23 21:27
"""
import codecs
import json
from torchtext import data
import torchtext.vocab as vocab
from utils import *
import logging
import pickle


class DuReader(object):
    """DuReader dataset loader"""
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
        # (for .pt)
        processed_dataset_path = data_path_process + "/torchtext/"
        train_examples_path = processed_dataset_path + f'{self.config["train_file"]}.pt'
        dev_examples_path = processed_dataset_path + f'{self.config["dev_file"]}.pt'
        test_examples_path = processed_dataset_path + f'{self.config["test_file"]}.pt'

        # judge if need to preprocess raw data files
        # if not os.path.exists(f'{data_path_process}/{self.config["train_file"]}l'):
        #     self.logger.info("preprocess train  data...")
        #     self.preprocess(f'{data_path}/{self.config["train_file"]}',
        #                     save_path=f'{data_path_process}/{self.config["train_file"]}l')
        #
        # if not os.path.exists(f'{data_path_process}/{self.config["dev_file"]}l'):
        #     self.logger.info("preprocess dev  data...")
        #     self.preprocess(f'{data_path}/{self.config["dev_file"]}',
        #                     save_path=f'{data_path_process}/{self.config["dev_file"]}l')
        #
        # if not os.path.exists(f'{data_path_process}/{self.config["test_file"]}l'):
        #     self.logger.info("preprocess test  data...")
        #     self.preprocess(f'{data_path}/{self.config["test_file"]}',
        #                     save_path=f'{data_path_process}/{self.config["test_file"]}l', train=False)

        # define Field
        self.logger.info("construct data loader....")
        self.RAW = data.RawField()
        self.RAW.is_target = False     # 读取id值
        self.CHRA_NESTING = data.Field(sequential=True, use_vocab=True, tokenize=list, lower=True)
        self.CHAR = data.NestedField(self.CHRA_NESTING, use_vocab=True, tokenize=lambda x: x)      # [b, seq_len, w_len]
        self.WORD = data.Field(sequential=True, use_vocab=True, batch_first=True,
                               tokenize=lambda x: x, lower=False, include_lengths=False)
        self.Q_WORD = data.Field(sequential=True, use_vocab=True, batch_first=True,
                                 tokenize=lambda x: x, lower=False, include_lengths=True)
        # for multi para  [b, para_num, seq_len] or [b, para_num, seq_len, w_len]
        self.PARAS = data.NestedField(self.WORD, use_vocab=True, tokenize=lambda x: x, include_lengths=True)
        self.LABEL = data.Field(sequential=False, use_vocab=False, unk_token=None)

        dict_fields = {'question_id': ('id', self.RAW),
                       # 'question': [('q_word', self.WORD), ('q_char', self.CHAR)],
                       'question': ('q_word', self.Q_WORD),
                       'question_type': ('question_type', self.RAW),
                       # 'paragraph': [('c_word', self.WORD), ('c_char', self.CHAR)],
                       'paragraphs': ('paras_word', self.PARAS),
                       's_idx': ('s_idx', self.LABEL),
                       'e_idx': ('e_idx', self.LABEL),
                       'yesno_answers': ('yesno_answers', self.RAW),
                       'answer_para_idx': ('answer_para_idx', self.LABEL)
        }

        list_fields = [('id', self.RAW), ('q_word', self.Q_WORD),
                       # ('q_char', self.CHAR),
                       ('question_type', self.RAW),
                       # ('c_word', self.WORD), ('c_char', self.CHAR),
                       ('paras_word', self.PARAS),
                       ('s_idx', self.LABEL), ('e_idx', self.LABEL),
                       ('yesno_answers', self.RAW),
                       ('answer_para_idx', self.LABEL)
                       ]

        test_dict_fields = {'question_id': ('id', self.RAW),
                            # 'question': [('q_word', self.WORD), ('q_char', self.CHAR)],
                            'question': ('q_word', self.Q_WORD),
                            'question_type': ('question_type', self.RAW),
                            # 'paragraph': [('c_word', self.WORD), ('c_char', self.CHAR)],
                            'paragraphs': ('paras_word', self.PARAS),
                            'yesno_answers': ('yesno_answers', self.RAW),
                            }

        test_list_fields = [('id', self.RAW), ('q_word', self.Q_WORD),
                            # ('q_char', self.CHAR),
                            ('question_type', self.RAW),
                            # ('c_word', self.WORD), ('c_char', self.CHAR),
                            ('paras_word', self.PARAS),
                            ('yesno_answers', self.RAW)
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

    def preprocess(self, path, save_path, train=True):
        """
        read preprocessed data or pre preprocessed data.
        :param path:
        :param save_path:
        :param train:
        :return:
        """
        if "v1.0" in self.config["data_path"]:
            self.logger.info("read processed data...")
            self.read_preprocess(path, save_path, train=train)
        elif "v2.0" in self.config["data_path"]:
            self.logger.info("pre preprocessed data...")
            self.pre_preprocess(path, save_path, train=train)

    def read_preprocess(self, path, save_path, train=True):
        """preprocess the process data to a list of dict. (just read)
            1. 使用预先处理的已经分词的数据
            2. 使用一个sample的处理之后字段
        ----------
        :param path:
        :param save_path:
        :return:
        (文本均是分词后的结果)
         train_d = {
            "question_id": "",
            "question": "",
            "question_type": "",
            "paragraph": "",
            "s_idx": 12,
            "e_idx": 13,
            "fake_answer": "",
            "yesno_answers": "",
            "match_score": 0.8  # the F1 of fake_answer and true answer
        }  # 训练一个找answer span的模型 + 判断yes_no, 可测试时候怎么做？
        """
        # read process data.
        datas = []
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if (idx + 1) % 1000 == 0:
                    self.logger.info("processed: {}".format(idx + 1))
                sample = json.loads(line.strip())
                # just pass for no answer sample.(for train)
                if train:
                    if len(sample["answers"]) == 0:
                        continue
                    if len(sample['answer_spans']) == 0:
                        continue
                    if sample["answer_spans"][0][0] < 0 or sample['answer_spans'][0][1] < 0:
                        continue
                # copy to data
                data = {}
                data["question_id"] = sample["question_id"]
                data["question"] = sample["segmented_question"]
                data["question_type"] = sample["question_type"]
                if "yesno_answers" in sample.keys():
                    data["yesno_answers"] = sample["yesno_answers"]
                else:
                    data["yesno_answers"] = []
                if train:
                    # find para
                    # sample["answer_docs"]有可能为空，即使answer不为空
                    if len(sample["answer_docs"]) == 0:
                        continue
                    answer_doc_idx = sample["answer_docs"][0]
                    # make sure answer doc idx in len(sample["documents"]
                    if answer_doc_idx not in range(len(sample["documents"])):
                        print("doc idx: ", answer_doc_idx,  "  len(documents): ", len(sample["documents"]))
                        continue
                    related_para_idx = sample["documents"][answer_doc_idx]["most_related_para"]
                    paragraph = sample["documents"][answer_doc_idx]["segmented_paragraphs"][related_para_idx]
                    data["paragraph"] = paragraph[: 500] if len(paragraph) >= 500 else paragraph
                    # find answer span
                    data["s_idx"], data["e_idx"] = sample["answer_spans"][0][0], sample["answer_spans"][0][1]
                    # make sure s_idx, e_idx is in [0, len(data["paragraph"]-1]
                    if data["e_idx"] > len(data["paragraph"]) - 1:
                        print("over len e_idx: ", data["e_idx"])
                        continue
                    data["match_score"] = sample["match_scores"][0]
                else:
                    # for test data to choose para
                    # 计算最佳的para
                    best_para = []
                    # 取最前面三个的不为空的document
                    docs = []
                    for doc in sample["documents"]:
                        if len(doc["segmented_paragraphs"]) != 0:
                            docs.append(doc)
                    docs = docs[: 3]
                    for doc in docs:
                        title = doc["segmented_title"]
                        # 取前面4个不为空的para
                        paras = []
                        for para in doc["segmented_paragraphs"]:
                            if len(para) != 0:
                                paras.append(para)
                        paras = paras[: 4]
                        # 全部拼接起来
                        best_para = best_para + title
                        for para in paras:
                            best_para = best_para + para
                    # 截取一定的长度(默认500)
                    best_para = best_para[: 500] if len(best_para) > 500 else best_para
                    data["paragraph"] = best_para
                    # pass the no para samples(special deal with)
                    if len(best_para) == 0:
                        continue
                datas.append(data)
        # write to processed data file
        self.logger.info("processed done! write to file!")
        with codecs.open(save_path, "w", encoding="utf-8") as f_out:
            for line in datas:
                json.dump(line, f_out, ensure_ascii=False)
                print("", file=f_out)

    def pre_preprocess(self, path, save_path, train=True):
        """preprocess the process data to a list of dict. (own preprocess method)
            1. 使用预先处理的已经分词的数据
            2. 使用一个sample的字段有：
                “question_id”: ,
                "question_type": ,
                "segmented_question": ,
                "documents": [
                                    ["segmented_title":   ,  "segmented_paragraphs": []] ，
                                    ["segmented_title":   ,  "segmented_paragraphs": []],
                                    ["segmented_title":   ,  "segmented_paragraphs": []].
                           ]
                "segmented_answers": [ ] ,
             3. 仅仅使用前三篇的document, 使用每个document的所有title+paragraph替换paragraph（保证截取文本的长度不超过预先设置的最大长度(500)）
             4. 计算各个paragraph和问题的BLUE-4分数，以衡量paragraph和问题的相关性，在分数前K的paragraph中，选择最早出现的paragraph.
              (paragraph选好了)
            5. 对于每个答案，在paragraph中选择与答案F1分数最高的片段，作为这个答案的参考答案片段；如果只有一个答案的模型，
            选择任意一个答案或者F1分数最高的那个答案对应的最佳的片段作为参考答案片段，训练时使用。
        ----------
        # question_type: "YES_NO": 0, "DESCRIPTION": 1, "ENTITY": 2
        # cyesno_answers: "Yes": 0, "No": 1, "Depends": "2"
        :param path:
        :return:
        (文本均是分词后的结果)
         train_d = {
            "question_id": "",
            "question": "",
            "question_type": "",
            "paragraphs": "[[], [], ..., []]",   # 多个para
            "s_idx": 12,
            "e_idx": 13,
            "fake_answer": "",
            "yesno_answers": "",
            "match_score": 0.8 , # the F1 of fake_answer and true answer
            "para_idx": 3   # 可用于训练para ranking
        }  # 训练一个找answer span的模型 + 判断yes_no, 可测试时候怎么做？
        """
        # read process data.
        datas = []
        with open(path, 'r', encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if (idx+1) % 1000 == 0:
                    self.logger.info("processed: {}".format(idx+1))
                sample = json.loads(line.strip())
                # just pass for no answer sample.(for train)
                if train:
                    if "answers" not in sample.keys() or len(sample["answers"]) == 0:
                        continue
                # copy to data
                data = {}
                data["question_id"] = sample["question_id"]
                data["question"] = sample["segmented_question"]
                data["question_type"] = sample["question_type"]
                if "yesno_answers" in sample.keys():
                    data["yesno_answers"] = sample["yesno_answers"]
                else:
                    data["yesno_answers"] = []
                # 过滤sample中paragraphs的非法字符
                sample = filter_illegal_words(sample)
                # find para (for zhidao and search)
                if "zhidao" in path:
                    best_paras = find_zhidao_paras(sample, train)
                elif "search" in path:
                    best_paras = find_search_paras(sample)
                else:
                    raise Exception("not supported data processing!")
                # skip len(best_paras)=0 samples
                if len(best_paras) == 0:
                    continue
                # choose best paras and  answer spans
                # data["paragraph"] = choose_one_para(best_paras, sample["segmented_question"], recall)  # 当前使用单para
                data["paragraphs"] = best_paras  # multiple paras
                if train:
                    data["fake_answer"], data["s_idx"], data["e_idx"], data["match_score"], data["answer_para_idx"] \
                        = find_fake_answer_from_multi_paras(sample, data["paragraphs"])
                # for paragraphs post_process(just for torchText easy to use)
                data["paragraphs"] = post_process_paras(best_paras, max_len=3)
                if not train or data['match_score'] != 0:
                    datas.append(data)
        # write to processed data file
        self.logger.info("processed done! write to file!")
        with codecs.open(save_path, "w", encoding="utf-8") as f_out:
            for line in datas:
                json.dump(line, f_out, ensure_ascii=False)
                print("", file=f_out)


if __name__ == "__main__":
    config = json.load(open('du_config.json', 'r'))
    # config['data_loader']['args']['data_path'] = '../data/dureader/raw'
    dureader = DuReader(config)
    for idx, data in enumerate(dureader.eval_iter):
        print("idx: ", idx, " ", data)
