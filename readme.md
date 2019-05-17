# MRC2019
- [2019百度机器阅读理解(Machine Reading Comprehension)](http://lic2019.ccf.org.cn/read)
- 参赛模型: BiDAF
- 最终排名:

## Develop Requirements
- OS: Ubuntu 16.04 LTS (64bit)
- GPU: Nvidia Titan Xp
- Language: Python 3.6.2.
- Pytorch: **0.4.1**
- torch==0.4.0
- nltk==3.2.4
- tensorboardX==0.8
- torchtext==0.3.1

## Data

类型 | train | dev | test
---|---|---|---|
[2019比赛官网数据下载](http://lic2019.ccf.org.cn/read) | 27W| 3000 | 7000 |

## Performance
Model | data type| zhidao_dev(Rouge-L) | zhidao_dev_(Bleu-4)|search_dev(Rouge-L)|search_dev(Blue-4)|**Rouge-L**|Blue-4
---|---|---|---|---|---|---|--- 
BiDAF | data_v1.0_指定para | 67.29 | |58.28| | 36.14 | 25.78|
BiDAF | process_3para_指定para | 56.78 | | | |   |  |
BiDAF | process_3para | 44.57 | | | |   |  |
BiDAF+pretrain_w2v | process_3para | 50.08 | |42.42| | 48.59 | 30.81 |
BiDAF+pretrain_w2v+yes_no | process_3para | 51.60 | |42.42| | 50.02 | 44.45 |
BiDAF+pretrain_w2v+PR+yes_no | process_3para | 52.42 | |42.72| |  50.41 | 45.16 |
BiDAF+pretrain_w2v+PR+yes_no(ensemble) | process_3para |  | | | | 50.78  | 44.53 |
R-net(too slow) |  |  | | | |   |  |


## competition process
1. 对zhidao和search进行数据预处理, 以启发式的方法尽可能的筛选过滤数据(花费了大部分时间)
2. 利用PyTorch模版，加载dureader数据，实现BiDAF
3. 使用w2v在zhidao和search数据上训练词向量，作为训练的初始向量
4. 在BiDAF中实现PR，实现multitask
5. 实验yes_no分类
6. 实现R-Net，并行训练
7. 多answers训练，加权loss  [TODO]
8. 先预测一个范围，在预测精准答案span  [TODO]












