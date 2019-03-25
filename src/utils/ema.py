#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: ema.py
@time: 2019/3/22 20:41
"""


class EMA(object):
    """ 按照mu来依次更新某个变量的平均值"""
    def __init__(self, mu):
        """
        :param mu: exp decay rate
        """
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def get(self, name):
        return self.shadow[name]

    def update(self, name, x):
        assert name in self.shadow
        new_average = (1.0 - self.mu) * x + self.mu * self.shadow[name]
        self.shadow[name] = new_average.clone()
