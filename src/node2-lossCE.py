#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A tensor flow script to classify a linearly separable dataset

@author: soren
"""
#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

filename_queue = tf.train.string_input_producer(['linear-data.csv'])
N = 5000
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
record_defaults = [[0.], [1.], [1.]]
col1, col2, col3 = tf.decode_csv(value, record_defaults=record_defaults)
features = tf.pack([col1, col2])
examples = np.empty([N, 2], dtype = float)
labels = np.empty([N, 2], dtype = int)
with tf.Session() as sess:
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    tf.initialize_all_variables().run()
    for i in range(N):
        # Retrieve a single instance:
        examples[i], labels[i,0] = sess.run([features, col3])
        if(labels[i,0] == 0):
            labels[i, 1] = 1
        else:
            labels[i,1] = 0
    coord.request_stop()
    coord.join(threads)
plt.scatter(examples[0:4000,0], examples[0:4000,1], c = labels[0:4000,0])
#%%
#Training Neural Net
x = tf.placeholder(tf.float32, [None, 2])
W = tf.Variable(tf.random_normal([2, 2], stddev = 0.35))
b = tf.Variable(tf.random_normal([1,2], stddev = 0.35))
y = tf.matmul(x, W) + b
y_ = tf.placeholder(tf.float32, [None, 2])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(prediction, "float"))
#%%
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    step = 10
    for i in range(0, 4000, step):
        sess.run(train_step, feed_dict={x: examples[i:i+step], y_: labels[i:i+step]})
    
    prediction_value = sess.run(tf.argmax(y,1), feed_dict={x:examples[4001:5000]})
    label_value = sess.run(tf.argmax(y_,1), feed_dict={y_:labels[4001:5000]})
    accuracy_value = sess.run(accuracy, feed_dict={x:examples[4001:5000], y_:labels[4001:5000]})
    W_value = W.eval()
    b_value = b.eval()  
print('Model Trained')
print('W = ' + repr(W_value))
print('b = ' + repr(b_value))  
print('Accuracy =' + repr(accuracy_value))
plt.scatter(examples[4001:5000,0], examples[4001:5000,1], c = prediction_value)