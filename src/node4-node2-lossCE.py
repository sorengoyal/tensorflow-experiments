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
filename_queue = tf.train.string_input_producer(['non-linear-data.csv'])
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
            labels[i, 1] = 0 
    coord.request_stop()
    coord.join(threads)
plt.scatter(examples[0:4000,0], examples[0:4000,1], c = labels[0:4000,0])

#%%
def init_weights(shape, init_method='xavier', xavier_params = (None, None)):
    if init_method == 'zeros':
        return tf.Variable(tf.zeros(shape, dtype=tf.float32))
    elif init_method == 'uniform':
        return tf.Variable(tf.random_normal(shape, stddev=0.01, dtype=tf.float32))
    else: #xavier
        (fan_in, fan_out) = xavier_params
        low = -4*np.sqrt(6.0/(fan_in + fan_out)) # {sigmoid:4, tanh:1} 
        high = 4*np.sqrt(6.0/(fan_in + fan_out))
        return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))
#%%
#Training Neural Net
num_hidden = 10
x = tf.placeholder(tf.float32, [None, 2])
W1 = init_weights([2, num_hidden], init_method = 'xavier', xavier_params = (2, num_hidden))#tf.Variable(tf.random_normal([2, num_hidden], stddev = 0.1))
b1 = init_weights([1, num_hidden], init_method = 'uniform')
z1 = tf.sigmoid(tf.matmul(x, W1) + b1)
W2 = init_weights([num_hidden, 2], init_method = 'xavier', xavier_params = (num_hidden, 2))#tf.Variable(tf.random_normal([num_hidden, 2], stddev = 0.1))
b2 = init_weights([1, 2], init_method = 'uniform')
y = tf.matmul(z1, W2) + b2
y_ = tf.placeholder(tf.float32, [None, 2])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(prediction, "float"))
#%%
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    step = 20
    for i in range(0, 4000, step):
        sess.run(train_step, feed_dict={x: examples[i:i+step], y_: labels[i:i+step]})
    
    prediction_value = sess.run(tf.nn.softmax(y), feed_dict={x:examples[4000:5000]})
    accuracy_value = sess.run(accuracy, feed_dict={x:examples[4000:5000], y_:labels[4000:5000]})
    #prediction_value = sess.run(tf.argmax(y,1), feed_dict={x:examples[14001:15000]})
    #accuracy_value = sess.run(accuracy, feed_dict={x:examples[14001:15000], y_:labels[14001:15000]})
    val = sess.run(y, feed_dict={x: [[0.5, 0.5], [0,0]]})
    W1_value = W1.eval()
    b1_value = b1.eval()  
    W2_value = W2.eval()
    b2_value = b2.eval()  
print('Model Trained')
print('W1 = ' + repr(W1_value))
print('b1 = ' + repr(b1_value))  
print('W2 = ' + repr(W2_value))
print('b2 = ' + repr(b2_value))  
print('Accuracy =' + repr(accuracy_value))
#plt.scatter(examples[14001:15000,0], examples[14001:15000,1], c = prediction_value)
plt.scatter(examples[4000:5000,0], examples[4000:5000,1], c = np.rint(prediction_value[:,0]))#, cmap = 'gray')