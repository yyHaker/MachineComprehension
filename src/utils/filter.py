#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: illegal_words.py
@time: 2019/4/9 10:39
"""


class IllegalWords(object):
    def __init__(self):
        self.illegal_words = [
            '<',
            '>',
            'p',
            'img',
            '=',
            'src',
            '\\',
            '/',
            '&',
            'nbsp',
            '★',
            '☆'
        ]