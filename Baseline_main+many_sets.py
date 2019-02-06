from preprocess_corpus import *
import tensorflow as tf
import json
import string
from nltk import word_tokenize
import numpy as np
from collections import Counter
from keras.preprocessing import sequence
import ENCODERmodule
import random
from statistics import mean
from skopt import gp_minimize
#TODO add more metrics



def neural(params):
    tf.reset_default_graph()
    #[lambda_, gamma_, lstm_units, lr, batch_size]
    lambda_, gamma_, shared_lstm_units, common_lstm_units, lr, batch_size = params
    maxlen = 160
    epochs = 5
    ADV = True
    names = ['Books', 'Electronics', 'Movies_and_TV', 'CDs_and_Vinyl', 'Clothing_Shoes_and_Jewelry', 'Home_and_Kitchen',
        'Kindle_Store', 'Sports_and_Outdoors', 'Cell_Phones_and_Accessories', 'Health_and_Personal_Care', 'Toys_and_Games',
        'Video_Games', 'Tools_and_Home_Improvement', 'Beauty', 'Apps_for_Android']
    #names = ['Books','Electronics']

    prob = tf.placeholder_with_default(1.0, shape=())

    if ADV:
        with tf.variable_scope("shared_lstm"):
            lstm_cell_s = tf.nn.rnn_cell.LSTMCell(shared_lstm_units, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123))  # tf.contrib.rnn.BasicLSTMCell
            lstm_cell_s = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell_s, output_keep_prob=prob) # FINALIZE FALLS HERE - WHY

    all_texts_train, all_texts_test, all_labels_train, all_labels_test = [], [], [], []
    task_pr = []
    for i in range(len(names)):
        texts_train, texts_test, labels_train, labels_test = preprocess(names[i], maxlen, remake=False)
        all_texts_train.append(texts_train)
        all_texts_test.append(texts_test)
        all_labels_train.append(labels_train)
        all_labels_test.append(labels_test)



        task_pr.append(task_private(names[i], common_lstm_units, shared_lstm_units, batch_size, maxlen, baseline=not ADV, prob=prob))

        if ADV:
            task_pr[i].value_s, _ = tf.nn.dynamic_rnn(lstm_cell_s, task_pr[i].lstm_feed, dtype=tf.float32)  # batch_size vectors with label "name"
            task_pr[i].trans_value_s = tf.transpose(task_pr[i].value_s, [1, 0, 2])
            task_pr[i].last_hidden_s = tf.gather(task_pr[i].trans_value_s, int(task_pr[i].trans_value_s.get_shape()[0]) - 1)
            task_pr[i].prediction = tf.matmul(tf.concat([task_pr[i].last_hidden_s, task_pr[i].last_hidden], 1), task_pr[i].weight) + task_pr[i].bias
            task_pr[i].diff_loss = diff_loss(task_pr[i].last_hidden_s, task_pr[i].last_hidden)
            task_pr[i].task_labels = tf.one_hot([i]*batch_size, len(names))
        else:
            task_pr[i].prediction = tf.matmul(task_pr[i].last_hidden, task_pr[i].weight) + task_pr[i].bias
        task_pr[i].correctPred = tf.equal(tf.argmax(task_pr[i].prediction, 1), tf.argmax(task_pr[i].labels, 1))
        task_pr[i].accuracy = tf.reduce_mean(tf.cast(task_pr[i].correctPred, tf.float32))
        task_pr[i].loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=task_pr[i].prediction, labels=task_pr[i].labels))

    #print("***all corpora are processed***")
    #print("adversarial mode:", ADV)
    #Discriminator
    if ADV:
        with tf.variable_scope("Discr"):
            input_d = flip_gradient(tf.concat([task.last_hidden_s for task in task_pr], axis=0))
            labels_d = tf.concat([task.task_labels for task in task_pr], axis=0)
            weight_d = tf.get_variable('weight', dtype='float32', initializer=tf.truncated_normal([shared_lstm_units, len(names)], seed=4))
            bias_d = tf.get_variable('bias', dtype='float32', initializer=tf.constant(0.1, shape=[len(names)]))
            prediction_d = (tf.matmul(input_d, weight_d) + bias_d)
            correctPred_d = tf.equal(tf.argmax(prediction_d, 1), tf.argmax(labels_d, 1))
            accuracy_d = tf.reduce_mean(tf.cast(correctPred_d, tf.float32))

            loss_d = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction_d, labels=labels_d))

    LOSS = sum([task.loss for task in task_pr])
    if ADV:
        LOSS += lambda_*sum([task.diff_loss for task in task_pr])
        LOSS += gamma_*loss_d

    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(LOSS)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.graph.finalize()

        min_num_of_str = min([len(t) for t in all_texts_train])
        #print(min_num_of_str)
        for j in range(epochs):
            i = 0
            while i + batch_size < min_num_of_str:
                sess.run(train_op, feed_dict={**dict(zip([t.input_ for t in task_pr], [t[i:i+batch_size] for t in all_texts_train])),
                                              **dict(zip([t.labels for t in task_pr], [t[i:i+batch_size] for t in all_labels_train])), **{prob:0.75}})
                i += batch_size

        min_num_of_str = min([len(t) for t in all_texts_test])
        #print(min_num_of_str)
        i = 0

        while i + batch_size < min_num_of_str:
            for j in range(len(names)):

                acc, loss  = sess.run((task_pr[j].accuracy, task_pr[j].loss), feed_dict={task_pr[j].input_:all_texts_test[j][i:i+batch_size], task_pr[j].labels:all_labels_test[j][i:i+batch_size]})
                task_pr[j].test_acc = np.append(task_pr[j].test_acc, acc)
                task_pr[j].test_loss = np.append(task_pr[j].test_loss, loss)

                #if i % 50 == 0:
                 #   print()

            i += batch_size

        #for j in range(len(names)):
            #print(names[j], "acc:", task_pr[j].test_acc.mean()*100, 'loss:', task_pr[j].test_loss.mean())

        #print("all mean acc:", np.asarray([task.test_acc.mean() for task in task_pr]).mean(), "all mean loss:", np.asarray([task.test_loss.mean() for task in task_pr]).mean())
        #if ADV:
        #    save_path = saver.save(sess, "checkpoints/model-adv.ckpt")
        #else:
        #    save_path = saver.save(sess, "checkpoints/model-no-adv.ckpt")
        mean_loss = np.asarray([task.test_loss.mean() for task in task_pr]).mean()

        sess.close()
    return mean_loss
    #lambda_, gamma_, shared_lstm_units, common_lstm_units, lr, batch_size = param


#res = gp_minimize(neural, dimensions = [(0.0001, 0.1), (0.001, 0.5), [8, 16,32,64], [8, 16,32,64], (0.0005, 0.005), [2, 4, 8,16,32]], n_calls=100, x0=[0.06562964579531903, 0.1, 32, 32, 0.0010217605642775806, 8])
#res = gp_minimize(neural, dimensions = [[0.1], [0.5], [8], [8, 16,32,64], (0.0005, 0.005), [2, 4, 8,16,32]], n_calls=100, x0= [0.1, 0.5, 8, 32, 0.0017737311505466684, 32])

#ADV [0.1, 0.001, 8, 8, 0.0005, 8] ACC: 0.8333334 ALL MEAN LOSS: 0.39033148
# #NON ADV [0.1, 0.5, 8, 8, 0.0005, 4] ACC: 0.8217172 ALL MEAN LOSS: 0.43277892

res = gp_minimize(neural, dimensions = [(0.05, 0.2), (0.0005, 0.005), [4, 6, 8, 16], [4,6,8, 16], (0.0003, 0.001), [2, 4, 8,16,32]], n_calls=100, x0=[0.1, 0.001, 8, 8, 0.0005, 8])

print('params', res.x, 'loss', res.fun)

#neural([0.1, 0.5, 8, 8, 0.0005, 4])