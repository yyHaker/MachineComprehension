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
import logging


class DuReader(object):
    """DuReader dataset loader"""
    def __init__(self, config):
        # logger
        self.logger = logging.getLogger('MC')
        # params
        self.config = config["data_loader"]["args"]
        # set path
        data_path = self.config["data_path"]
        processed_dataset_path = data_path + "/torchtext/"
        train_examples_path = processed_dataset_path + 'train_examples.pt'
        dev_examples_path = processed_dataset_path + 'dev_examples.pt'
        test_examples_path = processed_dataset_path + 'test_examples.pt'

        # judge if need to preprocess raw data files
        if not os.path.exists(f'{data_path}/{self.config["train_file"]}l'):
            self.logger.info("preprocess train  data...")
            self.preprocess(f'{data_path}/{self.config["train_file"]}')
        if not os.path.exists(f'{data_path}/{self.config["dev_file"]}l'):
            self.logger.info("preprocess dev  data...")
            self.preprocess(f'{data_path}/{self.config["dev_file"]}')
        if not os.path.exists(f'{data_path}/{self.config["test_file"]}l'):
            self.logger.info("preprocess test  data...")
            self.preprocess(f'{data_path}/{self.config["test_file"]}', train=False)

        # define Field
        self.logger.info("construct data loader....")
        self.RAW = data.RawField()
        self.RAW.is_target = False     # 读取id值
        self.CHRA_NESTING = data.Field(sequential=True, use_vocab=True, tokenize=list, lower=True)
        self.CHAR = data.NestedField(self.CHRA_NESTING, use_vocab=True, tokenize=lambda x: x)
        self.WORD = data.Field(sequential=True, use_vocab=True, batch_first=True,
                               tokenize=lambda x: x, lower=True, include_lengths=True)
        self.LABEL = data.Field(sequential=False, use_vocab=False, unk_token=None)

        dict_fields = {'question_id': ('id', self.RAW),
                       'question': [('q_word', self.WORD), ('q_char', self.CHAR)],
                       'question_type': ('question_type', self.RAW),
                       'paragraph': [('c_word', self.WORD), ('c_char', self.CHAR)],
                       's_idx': ('s_idx', self.LABEL),
                       'e_idx': ('e_idx', self.LABEL),
                       'yesno_answers': ('yesno_answers', self.RAW)
        }

        list_fields = [('id', self.RAW), ('q_word', self.WORD), ('q_char', self.CHAR),
                       ('question_type', self.RAW), ('c_word', self.WORD), ('c_char', self.CHAR),
                       ('s_idx', self.LABEL), ('e_idx', self.LABEL),
                       ('yesno_answers', self.RAW)
                       ]

        test_dict_fields = {'question_id': ('id', self.RAW),
                                   'question': [('q_word', self.WORD), ('q_char', self.CHAR)],
                                   'question_type': ('question_type', self.RAW),
                                   'paragraph': [('c_word', self.WORD), ('c_char', self.CHAR)],
                                   'yesno_answers': ('yesno_answers', self.RAW)
                            }

        test_list_fields = [('id', self.RAW), ('q_word', self.WORD), ('q_char', self.CHAR),
                       ('question_type', self.RAW), ('c_word', self.WORD), ('c_char', self.CHAR),
                       ('yesno_answers', self.RAW)
                       ]

        # judge if need to build dataSet
        if not os.path.exists(train_examples_path) or not os.path.exists(dev_examples_path):
            self.logger.info("build train dataSet....")
            self.train, self.dev = data.TabularDataset.splits(
                path=data_path,
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
                path=os.path.join(data_path, f'{self.config["test_file"]}l'),
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
        self.logger.info("build vocab....")
        self.CHAR.build_vocab(self.train, self.dev)
        self.WORD.build_vocab(self.train, self.dev)

        # load pretrained embeddings
        Vectors = vocab.Vectors(self.config["pretrain_emd_file"])
        self.WORD.vocab.load_vectors(Vectors)
        # just for call easy
        self.vocab_vectors = self.WORD.vocab.vectors

        # build iterators
        self.logger.info("building iterators....")
        self.train_iter, self.eval_iter = data.BucketIterator.splits(datasets=(self.train, self.dev),
                                                                     batch_sizes=[self.config["train_batch_size"], self.config["dev_batch_size"]],
                                                                     sort_key=lambda x: len(x.c_word),
                                                                     sort_within_batch=True,
                                                                     device=self.config["device"],
                                                                     shuffle=True)

        self.test_iter = data.BucketIterator(dataset=self.test,
                                             batch_size=16,
                                             sort_key=lambda x: len(x.c_word),
                                             sort_within_batch=True,
                                             device=self.config["device"],
                                             shuffle=True)


    def preprocess(self, path, train=True):
        """
        read preprocessed data or pre preprocessed data.
        :param path:
        :param train:
        :return:
        """
        if "v1.0" in self.config["data_path"]:
            self.logger.info("read processed data...")
            self.read_preprocess(path, train=train)
        elif "v2.0" in self.config["data_path"]:
            self.logger.info("pre preprocessed data...")
            self.pre_preprocess(path, train=train)

    def read_preprocess(self, path, train=True):
        """preprocess the process data to a list of dict. (just read)
            1. 使用预先处理的已经分词的数据
            2. 使用一个sample的处理之后字段
        ----------
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
                    if sample['answer_spans'][0][1] >= 500:
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
                    data["paragraph"] = sample["documents"][answer_doc_idx]["segmented_paragraphs"][related_para_idx]
                    # find answer span
                    data["s_idx"], data["e_idx"] = sample["answer_spans"][0][0], sample["answer_spans"][0][1]
                    data["match_score"] = sample["match_scores"][0]
                else:
                    # for test data how to choose paragraph(match question)
                    max_score = 0.
                    best_para = []
                    docs = sample["documents"][: 3]
                    for doc in docs:
                        for p_idx, para_tokens in enumerate(doc['segmented_paragraphs']):
                            score = metric_max_over_ground_truths(recall, para_tokens, sample["segmented_question"])
                            if score > max_score:
                                max_score = score
                                best_para = para_tokens
                    data["paragraph"] = best_para
                datas.append(data)
        # write to processed data file
        self.logger.info("processed done! write to file!")
        with codecs.open(f'{path}l', "w", encoding="utf-8") as f_out:
            for line in datas:
                json.dump(line, f_out)
                print("", file=f_out)

    def pre_preprocess(self, path, train=True):
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
                if (idx+1) % 100 == 0:
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
                # find para
                if 'zhidao' in path:
                    data["paragraph"] = self._find_zhidao_para(sample)
                elif 'search' in path:
                    data["paragraph"] = self._find_search_para(sample)
                else:
                    raise AssertionError('Invalid path or file name(should include zhidao or search)')

                # find answer span
                if train:
                    data["fake_answer"], data["s_idx"], data["e_idx"], data["match_score"] \
                        = self._find_fake_answer(sample, data["paragraph"])
                datas.append(data)
        # write to processed data file
        self.logger.info("processed done! write to file!")
        with codecs.open(f'{path}l', "w", encoding="utf-8") as f_out:
            for line in datas:
                json.dump(line, f_out)
                print("", file=f_out)

    def _find_zhidao_para(self, sample):
        """find zhidao paragraph.
        ----------
        1. 选择is_selected=True的前3个document
        2. 取每个document的前4个paragraph, 与title拼接， 用特殊符号<sep>分隔
        3. 拼接所有的paragraph, 截取不超过最大长度的内容（500词）
        :param sample:
        :return:
        """
        best_para = []
        docs = []
        for doc in sample["documents"]:
            if doc["is_selected"]:
                docs.append(doc)
        docs = docs[: 3]
        for doc in docs:
            title = doc["segmented_title"]
            paras = doc["segmented_paragraphs"][: 4]
            for para in paras:
                title = title.append("<sep>")
                para = title + para
                best_para = best_para + para
        # 截取一定的长度(默认500)
        best_para = best_para[: self.config['context_threshold']]
                # # assert len(para) >= 4 and len(sample["segmented_question"]) >= 4,\
                # #     print(sample["segmented_question"], ">>>>>>", print(para))
                # score = blue4(para, sample["segmented_question"])
                # result.append((para, score, idx))
        # sort by score
        # result = sorted(result, key=lambda x: x[1], reverse=True)
        # result = result[: top_k]
        # sort by idx
        # result = sorted(result, key=lambda x: x[2])
        # best_para, _, _ = result[0]
        return best_para

    def _find_search_para(self, sample):
        """ find search paragraph
        1. 选择is_select=True的前3篇doc，每个doc取前10个para
        2. 将标题和各段中间插入连接符号<sep>拼接在一起，没有超过最大长度（500）则返回这个段落，否则执行3
        3. 对于每个doc中的10个段落，计算各段落和问题的BLEU-4分数，来衡量段落和问题的相关性
        4. 在排名前3的段落中找到idx最小的段落（越靠前越有用），然后将该段落和该段落的下一段落拼接（下一段落可能包含答案）
        5. 在剩余8个段落中，每段选取第一句话
        6. 将上述所有内容拼接后，截取最大长度500返回
        :param sample:
        :return:
        """
        best_para = []
        # 选择前三个document
        docs = []
        for doc in sample["documents"]:
            if doc["is_selected"]:
                docs.append(doc)
        docs = docs[:3]
        for doc in docs:
            paras = doc["segmented_paragraphs"][:10]
            best_para += doc["segmented_title"]
            best_para += ['<sep>']
            for para in paras:
                best_para += para
        # 看全部拼接后长度是否超过阈值
        if len(best_para) <= 500:
            return best_para
        else:
            best_para = []
            question = sample['segmented_question']
            for doc in docs:
                # 拼title
                best_para += doc["segmented_title"]
                best_para += ['<sep>']
                # 仅选取前10个段落
                paras = doc["segmented_paragraphs"][:10]
                # 计算Recall
                prf_scores = [precision_recall_f1(para, question) for para in paras]
                scores = [i[1] for i in prf_scores]
                # 选取排名前2中最早出现的段落和下一段落
                scores_idx = [(i, scores[i]) for i in range(len(scores))]
                sorted_idx = sorted(scores_idx, key=lambda x: x[1], reverse=True)
                choose_idx = [i[0] for i in sorted_idx[:2]]
                # 拼接排名前2中最早出现的段落和下一段落
                early_idx = min(choose_idx)
                best_para += paras[early_idx]
                early_next_idx = early_idx + 1
                if early_next_idx < len(paras):
                    best_para += paras[early_next_idx]
                # 拼剩余段落的第一句话
                for i in sorted_idx:
                    best_para += self.first_sentence(paras[i[0]])
                    if len(best_para) > 500:
                        break
                # 截取最大长度500
                best_para = best_para[:500]
            return best_para

    @staticmethod
    def first_sentence(para):
        if not len(para):
            return []
        split_tag = ['。', '!', '?', '！', '？']
        s = []
        for word in para:
            s.append(word)
            if word in split_tag:
                break
        if s[-1] not in split_tag:
            s.append('。')
        return s




    def _find_fake_answer(self, sample, paragraph):
        """find paragraph and fake answer for one sample.
        :param samlpe:
        :param paragraph:
        :return:
        """
        best_para = paragraph
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
                        return best_para[best_start_idx: best_end_idx+1], best_start_idx, best_end_idx, best_score
        return best_para[best_start_idx: best_end_idx+1], best_start_idx, best_end_idx, best_score


if __name__ == "__main__":
    config = json.load(open('du_config.json', 'r'))
    # config['data_loader']['args']['data_path'] = '../data/dureader/raw'
    dureader = DuReader(config)
    for idx, data in enumerate(dureader.eval_iter):
        print("idx: ", idx, " ", data)
