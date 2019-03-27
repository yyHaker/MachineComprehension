#!/usr/bin/python
# coding:utf-8

"""
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


class DuReader(object):
    """DuReader dataset loader"""
    def __init__(self, config):
        # params
        self.config = config["data_loader"]["args"]
        # set path
        data_path = self.config["data_path"]
        processed_dataset_path = data_path + "/torchtext/"
        train_examples_path = processed_dataset_path + 'train_examples.pt'
        dev_examples_path = processed_dataset_path + 'dev_examples.pt'

        # judge if need to preprocess raw data files
        if not os.path.exists(f'{data_path}/{self.config["train_file"]}l'):
            print("preprocess raw data...")
            self.preprocess(f'{data_path}/{self.config["train_file"]}')
        if not os.path.exists(f'{data_path}/{self.config["dev_file"]}l'):
            self.preprocess(f'{data_path}/{self.config["dev_file"]}')

        # define Field
        print("construct data loader....")
        self.RAW = data.RawField()
        self.RAW.is_target = False     # 读取id值
        self.CHRA_NESTING = data.Field(sequential=True, use_vocab=True, tokenize=list, lower=True)
        self.CHAR = data.NestedField(self.CHRA_NESTING, use_vocab=True, tokenize=CN_tokenizer)
        self.WORD = data.Field(sequential=True, use_vocab=True, batch_first=True,
                               tokenize=CN_tokenizer, lower=True, include_lengths=True)
        self.LABEL = data.Field(sequential=False, use_vocab=False, unk_token=None)

        dict_fields = {'question_id': ('id', self.RAW),
                       'question': [('q_word', self.WORD), ('q_char', self.CHAR)],
                       'question_type': ('question_type', self.LABEL),
                       'paragraph': [('c_word', self.WORD), ('c_char', self.CHAR)],
                       's_idx': ('s_idx', self.LABEL),
                       'e_idx': ('e_idx', self.LABEL),
                       'yesno_answers': ('yesno_answers', self.LABEL)
        }

        list_fields = [('id', self.RAW), ('q_word', self.WORD), ('q_char', self.CHAR),
                       ('question_type', self.LABEL), ('c_word', self.WORD), ('c_char', self.CHAR),
                       ('s_idx', self.LABEL), ('e_idx', self.LABEL), ('yesno_answers', self.LABEL)]

        # judge if need to build dataSet
        if not os.path.exists(train_examples_path) or not os.path.exists(dev_examples_path):
            print("build dataSet....")
            self.train, self.dev = data.TabularDataset.splits(
                path=data_path,
                train=f'{self.config["train_file"]}l',
                validation=f'{self.config["dev_file"]}l',
                format='json',
                fields=dict_fields
            )
            # save preprocessed data
            os.mkdir(processed_dataset_path)
            torch.save(self.train.examples, train_examples_path)
            torch.save(self.dev.examples, dev_examples_path)

        else:
            print("loading dataSet.....")
            train_examples = torch.load(train_examples_path)
            dev_examples = torch.load(dev_examples_path)

            self.train = data.Dataset(examples=train_examples, fields=list_fields)
            self.dev = data.Dataset(examples=dev_examples, fields=list_fields)

        # cut too long context in the training set for efficiency
        if self.config["context_threshold"] > 0:
            self.train.examples = [e for e in self.train.examples if len(e.c_word) <= self.config["context_threshold"]]

        # build vocab
        self.CHAR.build_vocab(self.train, self.dev)
        self.WORD.build_vocab(self.train, self.dev)

        # load pretrained embeddings
        # Vectors = vocab.Vectors(self.config["pretrain_emd_file"])
        # self.WORD.vocab.load_vectors(Vectors)
        # # just for call easy
        # self.vocab_vectors = self.WORD.vocab.vectors

        # build iterators
        print("building iterators....")
        self.train_iter, self.eval_iter = data.BucketIterator.splits(datasets=(self.train, self.dev),
                                                                     batch_sizes=[self.config["train_batch_size"], self.config["dev_batch_size"]])

    def preprocess(self, path, train=True):
        """preprocess the raw data to a list of dict.
        1. read the raw data
        2. 计算最可能包含答案的paragraph：计算和q最相关的paragraph, 得到"ref_paragraph", "answer_docs",
        3. (train)计算answer span：计算每个答案(real answer)与paragraph的F1值，取最大的那个。得到"fake_answers", 计算answer span， "s_idx", "e_idx".
        4. "match_score": 计算fake answers和real answer的recall, 度量预处理得到fake_answer的效果
        ----------
        question_type: "YES_NO": 0, "DESCRIPTION": 1, "ENTITY": 2
        #cyesno_answers: "Yes": 0, "No": 1, "Depends": "2"
        :param path:
        :return:
         train_d = {
            "question_id": "",
            "question": "",
            "question_type": "",
            "paragraph": "",
            "s_idx": 12,
            "e_idx": 13,
            "fake_answer": "",
            "yesno_answers": "",
            "match_score": 0.8
        }  # 训练一个找answer span的模型 + 判断yes_no, 可测试时候怎么做？
        """
        # read raw data.
        datas = []
        with open(path, 'r', encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if (idx+1) % 1000 == 0:
                    print("processed: ", idx+1)
                sample = json.loads(line.strip())
                if len(sample["answers"]) == 0:
                    continue
                data = {}
                data["question_id"] = sample["question_id"]
                data["question"] = sample["question"]
                data["question_type"] = sample["question_type"]
                if "yesno_answers" in sample.keys():
                    data["yesno_answers"] = sample["yesno_answers"]
                else:
                    data["yesno_answers"] = []
                data["paragraph"], data["fake_answer"], data["s_idx"], data["e_idx"], data["match_score"] \
                    = self.find_para_fake_answer(sample)
                datas.append(data)
        # write to processed data file
        print("processed done! write to file!")
        with codecs.open(f'{path}l', "w", encoding="utf-8") as f_out:
            for line in datas:
                json.dump(line, f_out)
                print("", file=f_out)

    def find_para_fake_answer(self, sample):
        """find paragraph and fake answer for one sample.
        :param samlpe:
        :return:
        """
        # choose paragraph of the most related to question(recall)
        result = []
        for doc in sample["documents"]:
            for para in doc["paragraphs"]:   # not use doc["title]
                # TODO：change another method
                recall_v = metric_max_over_ground_truths(recall, para, sample["question"])
                result.append((para, recall_v))
        result = sorted(result, key=lambda x: x[1], reverse=True)
        best_para, _ = result[0]
        # 对于answers，在paragraph中找到与answers具有最大F1值的answer_span, [(answer, para, answer_span, F1)]
        # 取最大F1值的那个，作为"fake_answer", 得到"s_idx"、"e_idx"
        match_res = []
        for start_idx in range(len(best_para)):
            for end_idx in range(len(best_para)-1, start_idx, -1):
                span_string = best_para[start_idx: end_idx+1]
                F1_score = metric_max_over_ground_truths(f1_score, span_string, sample["answers"])
                match_res.append((start_idx, end_idx, F1_score))
        match_res = sorted(match_res, key=lambda x: x[2], reverse=True)
        best_start_idx, best_end_idx, best_score = match_res[0]
        return best_para, best_para[best_start_idx: best_end_idx+1], best_start_idx, best_end_idx, best_score


if __name__ == "__main__":
    config = json.load(open('du_config.json', 'r'))
    # config['data_loader']['args']['data_path'] = '../data/dureader/raw'
    dureader = DuReader(config)
    for idx, data in enumerate(dureader.eval_iter):
        print("idx: ", idx, " ", data)




