#! /usr/bin/bash
set -e
set -x

# cpu num
num=64
# define path
data_path="./data/dureader/process/2019/v2.0/zhidao.train.json"
# split_path="./data/dureader/process/2019/v2.0/split/zhidao.train.json"

# split the data
python process/divide.py $data_path $num

# load the split data to cope with
for ((i=0;i<$num;++i))
do
    python process/divid_process_zhidao.py ./data/dureader/process/2019/v2.0/split/zhidao.train.json__$i ./data/dureader/process/2019/v2.0/dealwith/zhidao.train.json__$i > ./process/log_$i 2>&1 &
done

# 能否设计成上面运行完成后，下面自动合并
# combine the result
# python process/combine.py ./data/dureader/process/2019/v2.0/dealwith/zhidao.train.json__ ./data/dureader/process/2019/v2.0/zhidao.train.jsonl 64

# for dev data
# python process/divid_process_zhidao.py ./data/dureader/process/2019/v2.0/split/zhidao.dev.json ./data/dureader/process/2019/v2.0/zhidao.train.jsonl
