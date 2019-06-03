#! /bin/sh

set -e
set -x

# divide zhidao data (store in split)
#python ./process/divide.py ./data/dureader/process/2019/v2.0/search.train.json 64

# split data to process
for ((i=0;i<64;++i))
do
    python ./process/divid_process_search.py ./data/dureader/process/2019/v2.0/split/search.train.json__$i ./data/dureader/process/2019/v2.0/divid_process/search.train.json__$i > ./data/dureader/process/2019/v2.0/split/log_search_$i 1 2>&1 &
donel

#for i in 0 19 20 23 29 56 57
#do
#    python ./process/divid_process_search.py ./data/dureader/process/2019/v2.0/split/search.train.json__$i ./data/dureader/process/2019/v2.0/divid_process/search.train.json__$i > ./data/dureader/process/2019/v2.0/split/log_search_$i 1 2>&1 &
#done


# combine result
#python ./process/combine.py ./data/dureader/process/2019/v2.0/divid_process/search.train.json__ ./data/dureader/process/2019/v2.0/search/three_para_multi_answers_filter/search.train.jsonl 64


# for dev
# python process/divid_process_search.py data/dureader/process/2019/v2.0/search.dev.json data/dureader/process/2019/v2.0/search/three_para_multi_answers_filter/search.dev.jsonl 1

# for test1
# python process/divid_process_search.py data/dureader/process/2019/v2.0/search.test1.json data/dureader/process/2019/v2.0/search/three_para_multi_answers_filter/search.test1.jsonl 0

