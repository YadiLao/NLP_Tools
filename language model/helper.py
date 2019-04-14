# !/usr/bin/python
# -*- coding:utf-8 -*-
# Author: yadi Lao
import math
import numpy as np
import operator
from functools import reduce, partial
import pickle
from sklearn.model_selection import train_test_split
from nltk.lm.preprocessing import pad_both_ends
import torch
from LM.utils import *


def prepare_data_ngram(data_path, test_size=0.3, n=3, random_state=42):
    """
    生成N_gram的训练测试数据 [(x_{i-n},...,x_{i-1}), x_i)]
    """
    data = load_data(data_path)
    data = [list(pad_both_ends(list(line.replace('\n', '')), 2)) for line in data[:5] if len(line) >= 2]

    print(data[:2])
    word_to_ix = generate_vocab(data)

    ngram_list = []
    ngram_list.append([(sen[i:i+n-1], sen[i + n-1]) for sen in data for i in range(len(sen) - 2)])

    print(ngram_list[:5])
    ngram_list = reduce(operator.add, ngram_list)
    print(ngram_list[:5])
    print(len(ngram_list))

    x_train, x_test = train_test_split(ngram_list, test_size=test_size, random_state=random_state)

    print(x_train[:3])
    print(x_test[:3])
    print('train={}, test={}'.format(len(x_train), len(x_test)))

    # indexing
    x_train, y_train = word2index(x_train, word_to_ix)
    x_test, y_test = word2index(x_test, word_to_ix)

    return x_train, x_test, y_train, y_test, word_to_ix


def prepare_data_cbow(data_path, test_size=0.3, n=3, random_state=42):
    """
    生成CBOW的训练测试数据， [(x_{i-n},...,x_{i-1}, x_{i+1},...,x_{i+n}), x_i)]
    """
    data = load_data(data_path)
    data = [list(pad_both_ends(list(line.replace('\n', '')), 2)) for line in data[:5] if len(line) >= 2]

    print(data[:2])
    word_to_ix = generate_vocab(data)

    ngram_list = []
    for sen in data:
        for i in range(n, len(sen)-n):
            context = sen[i-n:i] + sen[i+1:i+n+1]
            target = sen[i]
            ngram_list.append((context, target))

    print(ngram_list[:5])
    print(len(ngram_list))

    x_train, x_test = train_test_split(ngram_list, test_size=test_size, random_state=random_state)

    print(x_train[:3])
    print(x_test[:3])
    print('train={}, test={}'.format(len(x_train), len(x_test)))

    # indexing
    x_train, y_train = word2index(x_train, word_to_ix)
    x_test, y_test = word2index(x_test, word_to_ix)

    return x_train, x_test, y_train, y_test, word_to_ix


def generate_vocab(data):
    """
    生成词典
    """
    word_to_ix = defaultdict(int)
    word_to_ix['UNK'] = 0
    for line in data:
        for word in line:
            if word not in word_to_ix.keys():
                word_to_ix[word] = len(word_to_ix)

    print('vocab_size ={}'.format(len(word_to_ix)))

    return word_to_ix


def word2index(docs, vocab):
    """
    map word
    """
    x, y = [], []
    for doc in docs:
        context, target = doc[0], doc[1]
        context_idx = []
        for char in context:
            if char in vocab.keys():
                context_idx.append(vocab[char])
            else:
                context_idx.append(vocab['<UNK>'])

        target_idx = vocab[char] if char in vocab.keys() else vocab['UNK']

        x.append(context_idx)
        y.append([target_idx])

    return x, y


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    When used: batches = batch_iter(list(zip(train_x, train_y)), batch_size, epoch)
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            # If x is an integer, randomly permute np.arange(x).
            # If x is an array, make a copy and shuffle the elements randomly.
            # np.random.permutation(10)
            # array([1, 7, 4, 3, 0, 9, 2, 5, 8, 6])
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def test():
    import torch
    import torch.nn as nn
    m = nn.LogSoftmax()
    loss = nn.NLLLoss()
    # input is of size N x C = 3 x 5
    input = torch.randn(3, 5, requires_grad=True)
    # each element in target has to have 0 <= value < C
    target = torch.tensor([1, 0, 4])
    print(m(input).size(), target.size())
    output = loss(m(input), target)
    output.backward()


def n_gram_perplexity(probs):
    """
    计算N-Gram的perplexity
    tokens 为前后padding了<s>和</s>的list
    """
    p_s = [math.log(p) for p in probs]
    ps_sum = sum(p_s)

    perplexity = 2 ** (-1.0/len(p_s) * ps_sum)

    return perplexity


def plot_fig(loss, ppl, step_data, model_save_dir):

    if len(step_data) != len(loss) or len(step_data) != len(ppl):
        raise ValueError('Plot: length of x and y are not the same')

    fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax1.plot(step_data, loss, linewidth=3)
    ax1.set_title('train_loss')

    ax2.plot(step_data, ppl, linewidth=3)
    ax2.set_title('test ppl')
    plt.savefig(model_save_dir+'/train_fig.png', bbox_inches='tight')


def save_model(model, optimizer, path, epoch, loss):
    """
    save model
    """
    torch.save({
        'epoch': epoch, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)


def restore_model(model, optimizer, path):
    """
    restore
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model, optimizer, epoch, loss
    