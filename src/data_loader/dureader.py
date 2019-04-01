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
        self.CHAR = data.NestedField(self.CHRA_NESTING, use_vocab=True, tokenize=lambda x: x)
        self.WORD = data.Field(sequential=True, use_vocab=True, batch_first=True,
                               tokenize=lambda x: x, lower=True, include_lengths=True)
        self.LABEL = data.Field(sequential=False, use_vocab=False, unk_token=None)
        # self.q_LABEL = data.Field(sequential=False, use_vocab=True, unk_token=None)  # 可在预处理的时候做

        dict_fields = {'question_id': ('id', self.RAW),
                       'question': [('q_word', self.WORD), ('q_char', self.CHAR)],
                       'question_type': ('question_type', self.RAW),
                       'paragraph': [('c_word', self.WORD), ('c_char', self.CHAR)],
                       's_idx': ('s_idx', self.LABEL),
                       'e_idx': ('e_idx', self.LABEL)
                       # 'yesno_answers': ('yesno_answers', self.LABEL)
        }

        list_fields = [('id', self.RAW), ('q_word', self.WORD), ('q_char', self.CHAR),
                       ('question_type', self.RAW), ('c_word', self.WORD), ('c_char', self.CHAR),
                       ('s_idx', self.LABEL), ('e_idx', self.LABEL)]
                       # ('yesno_answers', self.LABEL)]

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
        # if self.config["context_threshold"] > 0:
        #     self.train.examples = [e for e in self.train.examples if len(e.c_word) <= self.config["context_threshold"]]

        # build vocab
        print("build vocab....")
        self.CHAR.build_vocab(self.train, self.dev)
        self.WORD.build_vocab(self.train, self.dev)
        # self.q_LABEL.build_vocab(self.train, self.dev)   # lable -> idx

        # load pretrained embeddings
        Vectors = vocab.Vectors(self.config["pretrain_emd_file"])
        self.WORD.vocab.load_vectors(Vectors)
        # just for call easy
        self.vocab_vectors = self.WORD.vocab.vectors

        # build iterators
        print("building iterators....")
        self.train_iter, self.eval_iter = data.BucketIterator.splits(datasets=(self.train, self.dev),
                                                                     batch_sizes=[self.config["train_batch_size"], self.config["dev_batch_size"]],
                                                                     sort_key=lambda x: len(x.c_word),
                                                                     sort_within_batch=True)

    def preprocess(self, path, train=True):
        """preprocess the process data to a list of dict.
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
        with open(path, 'r', encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if (idx+1) % 1000 == 0:
                    print("processed: ", idx+1)
                sample = json.loads(line.strip())
                # just pass for no answer sample.
                if len(sample["answers"]) == 0:
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
        top_docs_num = 3
        # para index(from 1)
        idx = 0
        top_k = 3
        result = []
        for doc in sample["documents"][:top_docs_num]:
            title = doc["segmented_title"]
            for para in doc["segmented_paragraphs"]:
                idx += 1
                para = title + para
                # 截取一定的长度(默认500)
                para = para[: 500] if len(para) > 500 else para
                # assert len(para) >= 4 and len(sample["segmented_question"]) >= 4,\
                #     print(sample["segmented_question"], ">>>>>>", print(para))
                score = blue4(para, sample["segmented_question"])
                result.append((para, score, idx))
        # sort by score
        result = sorted(result, key=lambda x: x[1], reverse=True)
        result = result[: top_k]
        # sort by idx
        result = sorted(result, key=lambda x: x[2])
        best_para, _, _ = result[0]
        # get answer tokens
        answer_tokens = set()
        for segmented_answer in sample["segmented_answers"]:
            answer_tokens = answer_tokens | set([token for token in segmented_answer])
        # choose answer span
        best_start_idx = 0
        best_end_idx = 0
        best_score = 0.
        for start_idx in range(len(best_para)):
            if best_para[start_idx] not in answer_tokens:
                continue  # speed the preprocess
            for end_idx in range(start_idx, len(best_para)):
                span_string = best_para[start_idx: end_idx+1]
                F1_score = metric_max_over_ground_truths(f1_score, span_string, sample["segmented_answers"])
                if F1_score > best_score:
                    best_score = F1_score
                    best_start_idx = start_idx
                    best_end_idx = end_idx
                    if F1_score == 1.0:
                        return best_para, best_para[best_start_idx: best_end_idx+1], best_start_idx, best_end_idx, best_score
        return best_para, best_para[best_start_idx: best_end_idx+1], best_start_idx, best_end_idx, best_score


if __name__ == "__main__":
    config = json.load(open('du_config.json', 'r'))
    # config['data_loader']['args']['data_path'] = '../data/dureader/raw'
    dureader = DuReader(config)
    for idx, data in enumerate(dureader.eval_iter):
        print("idx: ", idx, " ", data)




