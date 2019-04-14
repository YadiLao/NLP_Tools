# !/usr/bin/python
# -*- coding:utf-8 -*-
# Author: yadi Lao
from datetime importdate
import torch
import torch.autograd asautograd
import torch.nn asnn
import torch.nn.functional asF
import torch.optim asoptim
from LM.NeuralLM_helper import*

CONTEXT_SIZE = 2
EMBEDDING_DIM = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.01
EPOCH = 30
PATIENCE = 3


classNGramLanguageModeler(nn.Module):
    """
    N-Gram 语言模型
    """
    def__init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

        self.embeddings.weight.data.copy_(
            torch.from_numpy(self.random_embedding(vocab_size,embedding_dim)))

    def random_embedding(self, vocab_size, embedding_dim):pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0/ embedding_dim)
        forindex inrange(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def forward(self, inputs):embeds = self.embeddings(inputs).view((inputs.size()[0], -1))
        # print(embeds.size())out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out)
        return log_probs


class CBOW(nn.Module):
    """
    CBOW 语言模型
    """
    def__init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * 2* embedding_dim, 100)  # the only difference between NGramsself.linear2 = nn.Linear(100, vocab_size)

        self.embeddings.weight.data.copy_(
            torch.from_numpy(self.random_embedding(vocab_size,embedding_dim)))

    def random_embedding(self, vocab_size, embedding_dim):pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0/ embedding_dim)
        forindex inrange(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def forward(self, inputs):embeds = self.embeddings(inputs).view((inputs.size()[0], -1))
        # print(embeds.size())out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out)
        return log_probs


def train(model, x_train, y_train, x_test, y_test, model_save_dir, model_name='ngram'):
    """
    训练模型
    """
    step_num = int(math.ceil(len(x_train) * 1.0/ BATCH_SIZE))
    train_data = batch_iter(list(zip(x_train, y_train)), BATCH_SIZE, num_epochs=EPOCH, shuffle=True)

    loss_function = nn.NLLLoss()
    loss_list = []
    counter = 0
    plot_loss, plot_ppl, plot_step = [], [], []
    best_loss, best_ppl = 1e7, 1e7
    patience_step = 5
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    total_loss = torch.Tensor([0])

    t1 = time.time()
    for data in train_data:
        context, target = zip(*data)
        counter += 1
        context = np.array(context)

        target = list(reduce(operator.add, target))
        target = torch.tensor(target)

        # print(context.shape, target.size())
        model.zero_grad()
        log_probs = model(torch.from_numpy(context))
        #  Do the backward pass and update the gradientloss.backward()
        optimizer.step()
        total_loss += loss.data

        # ************* finish one episode *************
        if counter % step_num == 0:
            test_ppl = test(x_test, y_test, model)
            print('Epoch={}, loss={}, test_ppl={}, time={}'.format(
                counter / step_num, total_loss[0], test_ppl, time.time() - t1))
            iftest_ppl < best_ppl:
                save_model(model, optimizer, model_save_dir+'/'+model_name, counter/step_num, total_loss)
                patience_step = 0print('model save to {}!!!'.format(model_save_dir))
                best_ppl = test_ppl
            else:
                patience_step += 1
                if patience_step == PATIENCE:
                    break

            # *******************  plot train loss ***************plot_loss.append(total_loss)
            plot_ppl.append(test_ppl)
            plot_step.append(counter / step_num)
            plot_fig(plot_loss, plot_ppl, plot_step, model_save_dir)

            # resettotal_loss = torch.Tensor([0])
            t1 = time.time()

    loss_list.append(total_loss)
    print(loss_list)  # The loss decreased every iteration over the training data!returnloss_list


def test(x_test, y_test, model):
    """
    测试,并计算perplexity
    """
    test_data = batch_iter(list(zip(x_test, y_test)), BATCH_SIZE, num_epochs=1, shuffle=False)

    model.eval()
    perplexity_list = []

    for data in test_data:
        context, target = zip(*data)
        context = np.array(context)
        target = list(reduce(operator.add, target))
        log_probs = model(torch.from_numpy(context)).detach().numpy()
        ps = [log_probs[i][target[i]] fori inrange(len(log_probs))]

        # for i, prob in enumerate(log_probs.numpy()):#     ps.append(prob[target[i]])ps_sum = sum(ps)
        ppl = 2** (-1.0/ log_probs.shape[0] * ps_sum)
        perplexity_list.append(ppl)

    return sum(perplexity_list)/len(perplexity_list)


if__name__ == '__main__':
    data_path = '../data/label_corpus_extend.txt'
    model_name = 'ngram' # ngram or cbow

    time_stamp = str(date.today())
    model_save_dir = os.path.abspath(os.path.join('./runs', time_stamp))
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    if model_name == 'ngram':
        x_train, x_test, y_train, y_test, word_to_ix = prepare_data_ngram(data_path, n=CONTEXT_SIZE+1)
        vocab = word_to_ix.keys()
        model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
        for param in model.parameters():
            print(type(param.data), param.size())
        print(model)

    elif model_name == 'cbow':
        x_train, x_test, y_train, y_test, word_to_ix = prepare_data_cbow(data_path, n=CONTEXT_SIZE)
        vocab = word_to_ix.keys()
        model = CBOW(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
        for param in model.parameters():
            print(type(param.data), param.size())
        print(model)

    else:
        logging.error('No such model')
        x_train, x_test, y_train, y_test, word_to_ix, model = [None] * 6

    train(model, x_train, y_train, x_test, y_test, model_save_dir=model_save_dir, model_name=model_name)


