#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: logger.py
@time: 2019/3/9 15:42
"""
import json
import logging

logging.basicConfig(level=logging.INFO, format='')


class Logger(object):
    """
    Training process logger

    Note:
        Used by BaseTrainer to save training history.
    """
    def __init__(self):
        self.entries = {}

    def add_entry(self, entry):
        self.entries[len(self.entries) + 1] = entry

    def __str__(self):
        return json.dumps(self.entries, sort_keys=True, indent=4)
