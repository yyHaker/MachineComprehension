python test.py -m zhidao/BiDAFMultiParas/three_para_one_answer/model_best_Rouge_51.31.pth -d 1 -t ./result/predict/zhidao_result.json
python test.py -m search/BiDAFMultiParas/three_para_one_answer/model_best_Rouge_41.28.pth -d 1 -t ./result/predict/search_result.json
python combine_result.py -z ./result/predict/zhidao_result.json -s ./result/predict/search_result.json -t ./result/predict/result.json
