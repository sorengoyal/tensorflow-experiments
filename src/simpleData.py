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
filename_queue = tf.train.string_input_producer(['../datasets/linearData.csv'])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defaults = [[0.], [1.], [1.]]
x, y, t = tf.decode_csv(value, record_defaults=record_defaults)
features = tf.pack([x, y])
examples = np.empty([1000, 2], dtype = float)
labels = np.empty([1000, 1], dtype = int)
with tf.Session() as sess:
  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(1000):
    # Retrieve a single instance:
    examples[i], labels[i] = sess.run([features, t])

  coord.request_stop()
  coord.join(threads)

  
plt.scatter(examples[:,0], examples[:,1], c = labels)