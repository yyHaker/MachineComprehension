#! /bin/sh

set -e
set -x

# divide zhidao data (store in split)
# python ./process/divide.py ./data/dureader/process/2019/v2.0/zhidao.train.json 64

# split data to process
for ((i=0;i<64;++i))
do
    python ./process/divid_process_zhidao.py ./data/dureader/process/2019/v2.0/split/zhidao.train.json__$i ./data/dureader/process/2019/v2.0/divid_process/zhidao.train.json__$i > ./data/dureader/process/2019/v2.0/split/log_zhidao_$i 1 2>&1 &
done


# combine result
#python ./process/combine.py ./data/dureader/process/2019/v2.0/divid_process/zhidao.train.json__ ./data/dureader/process/2019/v2.0/zhidao/three_para_multi_answers_filter/zhidao.train.jsonl 64

# for test1
# python process/divid_process_zhidao.py data/dureader/process/2019/v2.0/zhidao.test1.json data/dureader/process/2019/v2.0/zhidao/three_para_multi_answers_filter/zhidao.test1.jsonl 0

# for dev data
# python process/divid_process_zhidao.py data/dureader/process/2019/v2.0/zhidao.dev.json data/dureader/process/2019/v2.0/zhidao/three_para_multi_answers_filter/zhidao.dev.jsonl 1

#
