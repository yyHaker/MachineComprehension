#! /usr/bin/bash
set -e
set -x

for ((i=0;i<64;++i))
do
    python divid_process_zhidao.py ./preprocess/search.train.json__$i ./preprocess/divid_pre/search.train.json__$i > ./preprocess/log_$i 2>&1 &
done

