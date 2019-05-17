#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: postprocess.py
@time: 2019/4/10 10:25
"""


def post_process_paras(paragraphs, max_len=3):
    """
    add special token to empty paras.
    :param paragraphs: composed of paragraph, paragraph contain several little paras.
    :param max_len: the max num of paragraph.
    :return:
    -------
    example:
    >>> a = [ ['me', 'hate', 'tes'],
                ['my']]
    >>> post_process_paras(a, max_len=4)
            [ ['me', 'hate', 'tes', '<eop>'],
                ['my', '<eop>'],
                ['<eop>']]
    """
    new_paras = []
    for paragraph in paragraphs:
        if len(paragraph) != 0:
            paragraph = paragraph + ["<eop>"]
            new_paras.append(paragraph)
    pad = ["<eop>"]
    c_len = len(new_paras)
    while c_len < max_len:
        new_paras.append(pad)
        c_len += 1
    return new_paras


def post_process_paras_flags(paragraphs, max_len=3):
    """
        add special token to every paragraph.
        :param paragraphs: composed of paragraph, paragraph contain several little paras.
        :param max_len: the max num of paragraph.
        :return:
        -------
        example:
        # 每个paragraph第一个<sep>前面的是title.
        >>> a = [
            ['<doc_0>', '<title>', title, '<sep>', p0, '<sep>', p1, '<sep>', p2],
            ['<doc_1>', '<title>', title, '<sep>', p0],
        ]
        >>> post_process_paras_flags(a, max_len=3)
        [
            ['<doc_0>', '<title>', title,'<para_0>', p0,'<para_1>', p1, '<para_2>', p2, '<eop>'],
            ['<doc_1>', '<title>', title, '<para_0>', p0, '<eop>'],
            ['empty']
        ]
        """
    new_paras = []
    for idx, paragraph in enumerate(paragraphs):
        if len(paragraph) != 0:
            tmp = _change_sep(paragraph)
            paragraph = tmp + ["<eop>"]
            new_paras.append(paragraph)
    pad = ["<empty>"]
    c_len = len(new_paras)
    while c_len < max_len:
        new_paras.append(pad)
        c_len += 1
    return new_paras


def _change_sep(paragraph):
    """
    change the flag <sep> in paragraph to <para_idx>.
    :param paragraph: composed of several paras.
    :return:
    ------
      example:
        # 每个paragraph第一个<sep>前面的是title.
    >>> a = [title, '<sep>', p0, '<sep>', p1, '<sep>', p2]
    >>> b = _change_sep(paragraph)
        [title, '<para_0>', p0, '<para_1>', p1, '<para_2>', p2]
    """
    count = 0
    for idx, ch in enumerate(paragraph):
        if ch == "<sep>":
            paragraph[idx] = "<para_{}>".format(count)
            count += 1
    return paragraph


if __name__ == "__main__":
    # a = [['me', 'hate', 'tes'],
    #             ['what', 'is'],
    #             ['my']]
    # print(post_process_paras(a, max_len=4))

    a = [
        ['title', '<sep>', 'p0', '<sep>', 'p1', '<sep>', 'p2'],
        ['title', '<sep>', 'p0'],
    ]
    b = post_process_paras_flags(a, max_len=3)
    print(b)

    # d = ['title', '<sep>', 'p0', '<sep>', 'p1', '<sep>', 'p2']
    # f = _change_sep(d)
    # print(f)

