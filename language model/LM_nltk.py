# !/usr/bin/python
# -*- coding:utf-8 -*-
# Author: yadi Lao
import os
import re
import codecs
import logging
import time

from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from LM.utils import *


def train_model(text, n_gram=3):
    """
    MLE训练基于统计的语言模型
    :param text: [['a','b'],['a','b','c']]
    """
    print('train size={}'.format(len(text)))
    train_data, padded_sents = padded_everygram_pipeline(n_gram, text)

    # train model
    model = MLE(n_gram)  # Lets train a 3-grams model, previously we set n=3
    model.fit(train_data, padded_sents)
    print('词表大小={}'.format(len(model.vocab)))

    return model


def calcalute_prob(model):
    """
    计算概率
    """
    print(model.counts['language'])                    # i.e. Count('language')
    print(model.counts[['language', 'is']]['never'])   # i.e. Count('never'|'language is')
    print(model.score('language'))                     # P('language')
    print(model.score('is', 'language'.split()))       # P('is'|'language')
    print(model.score('never', 'language is'.split()))  # P('never'|'language is')
    print(model.logscore("never", "language is".split()))


def generate_sent(model, num_words, random_seed=42):
    """
    :param model: An ngram language model from `nltk.lm.model`.
    :param num_words: Max no. of words to generate.
    :param random_seed: Seed value for random.
    """
    content = []
    for token in model.generate(num_words, random_seed=random_seed):
        if token == '<s>':
            continue
        if token == '</s>':
            break
        content.append(token)

    detokenize = TreebankWordDetokenizer().detokenize
    return detokenize(content)


def model_generate(model):
    print(model.generate(20, random_seed=7))


if __name__ == '__main__':
    data_path = '../data/label.txt'
    segData = load_seg_data(data_path)
    t1 = time.time()
    model = train_model(segData, n_gram=3)
    print('take {} times'.format(time.time()-t1))
    
    
