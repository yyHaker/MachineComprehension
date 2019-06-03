#!/usr/bin/env bash

# for zhidao toy data
 python process/divid_process_zhidao.py data/dureader/process/2019/v2.0/zhidao.dev.json data/dureader/process/2019/v2.0/zhidao/toydata/zhidao.dev_part.jsonl 1
 python process/divid_process_zhidao.py data/dureader/process/2019/v2.0/zhidao.train.json data/dureader/process/2019/v2.0/zhidao/toydata/zhidao.train_part.jsonl 1
 python process/divid_process_zhidao.py data/dureader/process/2019/v2.0/zhidao.test1.json data/dureader/process/2019/v2.0/zhidao/toydata/zhidao.test1_part.jsonl 0

# for search toy data
 python process/divid_process_search.py data/dureader/process/2019/v2.0/search.dev.json data/dureader/process/2019/v2.0/search/toydata/search.dev_part.jsonl 1
 python process/divid_process_search.py data/dureader/process/2019/v2.0/search.train.json data/dureader/process/2019/v2.0/search/toydata/search.train_part.jsonl 1
 python process/divid_process_search.py data/dureader/process/2019/v2.0/search.test1.json data/dureader/process/2019/v2.0/search/toydata/search.test1_part.jsonl 0