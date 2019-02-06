import json
import string
from nltk import word_tokenize
import numpy as np
from collections import Counter
from keras.preprocessing import sequence
import random
import os.path
import tensorflow as tf
from tensorflow.python.framework import ops

def diff_loss(A, B):
    return tf.norm(tf.matmul(tf.transpose(tf.to_float(A)), tf.to_float(B)))**2


class FlipGradientBuilder(object):
    '''Gradient Reversal Layer from https://github.com/pumpikano/tf-dann'''

    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y


flip_gradient = FlipGradientBuilder()

def preprocess(name, maxlen, remake = False):
    train_size = 1400
    test_size = 400
    trim_dict = 20000
    #maxlen = 160
    if not remake and os.path.isfile("/home/boeing/PycharmProjects/CNN/AMTL_" + name + "texts_train.npy") \
        and os.path.isfile("/home/boeing/PycharmProjects/CNN/AMTL_" + name + "texts_test.npy") \
        and os.path.isfile("/home/boeing/PycharmProjects/CNN/AMTL_" + name + "labels_train.npy") \
        and os.path.isfile("/home/boeing/PycharmProjects/CNN/AMTL_" + name + "labels_test.npy"):

        texts_train = np.load("/home/boeing/PycharmProjects/CNN/AMTL_" + name + "texts_train.npy")
        texts_test = np.load("/home/boeing/PycharmProjects/CNN/AMTL_" + name + "texts_test.npy")
        labels_train = np.load("/home/boeing/PycharmProjects/CNN/AMTL_" + name + "labels_train.npy")
        labels_test = np.load("/home/boeing/PycharmProjects/CNN/AMTL_" + name + "labels_test.npy")

        return texts_train, texts_test, labels_train, labels_test
    else:
        rews = []
        i = 0
        for line in open("/home/boeing/Downloads/" + name + "_5.json", "r"):
            if i < 20000:
                rews.append(json.loads(line))
                i += 1
            else:
                break

        texts = []
        labels = []

        '''pos, neg = 0, 0

        for i in range(len(rews)):
            if rews[i]['overall'] != 3:
                if rews[i]['overall'] > 3:
                    pos += 1
                if rews[i]['overall'] < 3:
                    neg += 1'''

        #balance_length = min(pos, neg)
        balance_length = (train_size + test_size)/2
        pos, neg = 0, 0

        #shuffle deterministically
        #random.Random(123).shuffle(rews)

        for i in range(len(rews)):
            if rews[i]['overall'] != 3:
                if rews[i]['overall'] > 3 and pos < balance_length:
                    texts.append([word.lower() for word in word_tokenize(rews[i]['reviewText']) if word.isalpha()])
                    labels.append([0, 1])
                    pos += 1
                if rews[i]['overall'] < 3 and neg < balance_length:
                    texts.append([word.lower() for word in word_tokenize(rews[i]['reviewText']) if word.isalpha()])
                    labels.append([1, 0])
                    neg += 1

        #print(texts[0:10], labels[0:10])

        #shuffle texts and labels the same way to cut off test later
        c = list(zip(texts, labels))
        random.Random(123).shuffle(c)
        texts, labels = zip(*c)
        texts, labels = list(texts), list(labels)

        glove_words = []
        glove_vecs = []
        i = 0
        #for line in open("/home/boeing/Downloads/glove.6B.200d.txt"):
        for line in open("/home/boeing/Downloads/glove.6B.50d.txt"):
            if i < trim_dict:
                splitted = line.split()
                if splitted[0].isalpha():
                    glove_words.append(splitted[0])
                    glove_vecs.append(splitted[1:])
                i += 1
        #print(glove_words[0:50])

        vocab_to_int = {}
        int_to_vocab = {}
        #word_counts = Counter(all_words)

        #unique_sorted_words = np.asarray(
        #    [word for word in sorted(word_counts, key=lambda k: word_counts[k], reverse=True)])

        #unique_sorted_words = unique_sorted_words[:trim_dict]

        #unique = set(unique_sorted_words)
        for i, word in enumerate(glove_words): #unique_sorted_words
            vocab_to_int[word] = i + 1 #i + 1
            int_to_vocab[i + 1] = word #i + 1 ZERO IS RESERVED FOR UNKNOWN
        #print([vocab_to_int[t] for t in ['the', 'good', 'stars', 'upset', 'quality']])



        glove_words = set(glove_words)

        for i in range(len(texts)):
            texts[i] = [vocab_to_int[word] for word in texts[i] if word in glove_words]
        #print(texts[0:10], labels[0:10])

        texts_train = np.asarray(texts[test_size : test_size + train_size])
        labels_train = np.asarray(labels[test_size : test_size + train_size])

        texts_test = np.asarray(texts[:test_size])
        labels_test = np.asarray(labels[:test_size])

        texts_train = sequence.pad_sequences(texts_train, maxlen=maxlen)
        texts_test = sequence.pad_sequences(texts_test, maxlen=maxlen)
        #print(texts_test[0:10], labels_test[0:10])
        np.save("/home/boeing/PycharmProjects/CNN/glove_emb", np.asarray(glove_vecs))
        np.save("/home/boeing/PycharmProjects/CNN/AMTL_" + name + "texts_train", texts_train)
        np.save("/home/boeing/PycharmProjects/CNN/AMTL_" + name + "texts_test", texts_test)
        np.save("/home/boeing/PycharmProjects/CNN/AMTL_" + name + "labels_train", labels_train)
        np.save("/home/boeing/PycharmProjects/CNN/AMTL_" + name + "labels_test", labels_test)
        print('*** preprocessed', name, len(texts_train), len(texts_test), "***")

        return texts_train, texts_test, labels_train, labels_test

'''def get_task_private_lstm(name, lstm_units, batch_size, maxlen, baseline):

    with tf.variable_scope(name):
        input_ = tf.placeholder(tf.int32, [batch_size, maxlen])
        labels = tf.placeholder(tf.int32, [batch_size, 2])
        emb_init = np.load("/home/boeing/PycharmProjects/CNN/glove_emb.npy")
        emb_init = np.concatenate((np.asarray([[0.0]*200]), emb_init), axis = 0)
        #print(emb_init.shape)
        emb_const = tf.constant(emb_init, dtype='float32', shape = emb_init.shape)
        embedding = tf.get_variable('embedding', dtype='float32', initializer=emb_const)

        #embedding = tf.get_variable('embedding', dtype='float32', initializer=tf.random_uniform((20001, 200), -1, 1, seed=123))
        lstm_feed = tf.nn.embedding_lookup(embedding, input_)

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_units) #tf.contrib.rnn.BasicLSTMCell tf.nn.rnn_cell.LSTMCell
        lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.75)
        value, _ = tf.nn.dynamic_rnn(lstm_cell, lstm_feed, dtype=tf.float32)
        if not baseline:
            weight = tf.get_variable('weight', dtype='float32', initializer=tf.truncated_normal([2*lstm_units, 2]))
        else:
            weight = tf.get_variable('weight', dtype='float32', initializer=tf.truncated_normal([lstm_units, 2]))
        bias = tf.get_variable('bias', dtype='float32', initializer=tf.constant(0.1, shape=[2]))

        value = tf.transpose(value, [1, 0, 2]) #[maxlen, batch, units]
        last_hidden = tf.gather(value, int(value.get_shape()[0]) - 1) #last hidden vector

        return input_, lstm_feed, labels, last_hidden, weight, bias'''

class task_private():
    def __init__(self, name, common_lstm_units, shared_lstm_units, batch_size, maxlen, baseline, prob):
        with tf.variable_scope(name):
            self.input_ = tf.placeholder(tf.int32, [batch_size, maxlen])
            self.labels = tf.placeholder(tf.int32, [batch_size, 2])
            emb_load = np.load("/home/boeing/PycharmProjects/CNN/glove_emb.npy")
            emb_init = np.concatenate((np.asarray([[0.0] * emb_load.shape[1]]), emb_load), axis=0)
            #print(emb_init.shape)
            self.emb_const = tf.constant(emb_init, dtype='float32', shape=emb_init.shape)
            self.embedding = tf.get_variable('embedding', dtype='float32', initializer=self.emb_const)

            # embedding = tf.get_variable('embedding', dtype='float32', initializer=tf.random_uniform((20001, 200), -1, 1, seed=123))
            self.lstm_feed = tf.nn.embedding_lookup(self.embedding, self.input_)

            self.lstm_cell = tf.nn.rnn_cell.LSTMCell(common_lstm_units, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123))  # tf.contrib.rnn.BasicLSTMCell tf.nn.rnn_cell.LSTMCell
            self.dropout_lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=self.lstm_cell, output_keep_prob=prob)
            self.value, _ = tf.nn.dynamic_rnn(self.dropout_lstm_cell, self.lstm_feed, dtype=tf.float32)
            if not baseline:
                self.weight = tf.get_variable('weight', dtype='float32',
                                              initializer=tf.truncated_normal([common_lstm_units+shared_lstm_units, 2], seed=4))
            else:
                self.weight = tf.get_variable('weight', dtype='float32', initializer=tf.truncated_normal([common_lstm_units, 2], seed=4))
            self.bias = tf.get_variable('bias', dtype='float32', initializer=tf.constant(0.1, shape=[2]))

            self.trans_value = tf.transpose(self.value, [1, 0, 2])  # [maxlen, batch, units]
            self.last_hidden = tf.gather(self.trans_value, int(self.trans_value.get_shape()[0]) - 1)
            self.test_acc = np.array([], dtype='float32')# last hidden vector
            self.test_loss = np.array([], dtype='float32')  # last hidden vector



#texts_train, texts_test, labels_train, labels_test = preprocess("Books")
#print(texts_train[0:3], labels_train[0:3], len(texts_test), len(texts_train))
#get_task_private_lstm('Books', 32)