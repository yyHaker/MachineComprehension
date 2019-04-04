python test.py -m zhidao/BiDAF/model_best_Rouge_67.29.pth -d 0 -t ./result/predict/zhidao_result.json
python test.py -m search/BiDAF/model_best_Rouge_58.28.pth -d 0 -t ./result/predict/search_result.json
python combine_result.py -z ./result/predict/zhidao_result.json -s ./result/predict/search_result.json -t ./result/predict/result.json