# !/usr/bin/python
# -*- coding:utf-8 -*-
# 基于拼音的模糊匹配
# Author: yadi Lao

import sys, os
import logging
import re
import operator
from functools import reduce
from pypinyin import pinyin, lazy_pinyin, Style
logging.basicConfig(level=logging.DEBUG)


# 拼音的分隔符
py_space = "\\s|\\.|\\,|\\!|\\?|。|，|！|？"
# 模糊匹配配置
fuzzy_start = [
    ('sh', 'ch', 'zh'),
    ('s', 'c', 'z'),
]
fuzzy_end = [
    ('ang', 'eng', 'ing'),
    ('an', 'en', 'in'),
]

match_source_map = {-1: 'Not Matched', 0: 'full', 1: 'pinyin', 2: 'pinyin_fuzzy'}


def full_match(match_str, str):
    """
    完全匹配拼音声调
    """
    str_ls = reduce(operator.add, pinyin(str))
    match_ls = reduce(operator.add, pinyin(match_str))

    if py_full_math(match_ls, str_ls):
        return 0, match_str
    else:
        return -1, 'None'


def pinyin_match(match_str, str):
    """
    拼音匹配，忽略声调
    """
    str_ls = lazy_pinyin(str)
    match_ls = lazy_pinyin(match_str)
    if py_full_math(match_ls, str_ls):
        return 1, match_str
    else:
        return -1, 'None'


def pinyin_fuzzy_match(match_str, str):
    """
    拼音模糊匹配，忽略声调和易混淆发音
    """
    str_ls = lazy_pinyin(str)
    match_ls = lazy_pinyin(match_str)
    f_str_ls = py_fuzzy_format(str_ls)
    f_match_ls = py_fuzzy_format(match_ls)
    if py_full_math(f_match_ls, f_str_ls):
        return 2, match_str
    else:
        return -1, 'None'


def py_full_math(match_ls, str_ls):
    """
    拼音字符串是否匹配
    """
    str_py = ' '.join(str_ls)
    match_py = ' '.join(match_ls)
    if re.findall("(^|{start}){match}($|{end})".format(start=py_space, match=match_py, end=py_space,), str_py):
        return True
    return False


def py_fuzzy_format(py_ls):
    """
    格式化模糊拼音
    """
    new_py_ls = []
    # 开始匹配
    for word in py_ls:
        # 前缀匹配
        match_word = None
        for match, rep in zip(fuzzy_start[0], fuzzy_start[1]):
            if word.startswith(match):
                match_word = (match, rep)
                break
        if match_word:
            word = match_word[1] + word[len(match_word[0]):]

        # 后缀匹配
        match_word = None
        for match, rep in zip(fuzzy_end[0], fuzzy_end[1]):
            if word.endswith(match):
                match_word = (match, rep)
                break
        if match_word:
            word = word[:-len(match_word[0])] + match_word[1]

        new_py_ls.append(word)

    return new_py_ls


def test_match():
    """
    test 三种匹配模式
    """
    word = '我们一起去六分三看看吧'
    redWord = '六峰山'
    print(full_match(redWord, word))
    print(pinyin_match(redWord, word))
    print(pinyin_fuzzy_match(redWord, word))


if __name__ == '__main__':
    test_match()





