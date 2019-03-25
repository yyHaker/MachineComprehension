# BiDAF-pytorch
Re-implementation of [BiDAF](https://arxiv.org/abs/1611.01603)(Bidirectional Attention Flow for Machine Comprehension, Minjoon Seo et al., ICLR 2017) on PyTorch.

## Results

Dataset: [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/)

| Model(Single) | EM(%)(dev) | F1(%)(dev) |
|--------------|:----------:|:----------:|
| **Re-implementation** | **65.3** | **75.3** | 
| Baseline(paper) | 67.7 | 77.3 |

## Development Environment
- OS: Ubuntu 16.04 LTS (64bit)
- GPU: Nvidia Titan Xp
- Language: Python 3.6.2.
- Pytorch: **0.4.0**

## Requirements

Please install the following library requirements specified in the **requirements.txt** first.

    torch==0.4.0
    nltk==3.2.4
    tensorboardX==0.8
    torchtext==0.3.1

## Execution

> python train.py 

