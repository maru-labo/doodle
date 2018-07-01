# coding: utf-8
# Copyright (c) 2018 一般社団法人 MaruLabo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
    Juefei-Xu, Felix, Vishnu Naresh Boddeti, and Marios Savvides.
    "Local binary convolutional neural networks." Computer Vision and Pattern
    Recognition (CVPR), 2017 IEEE Conference on. Vol. 1. IEEE, 2017.
"""

import tensorflow as tf

def bernoulli(shape):
    B = tf.distributions.Bernoulli(0.5)
    sparsity = B.sample(shape)
    return tf.to_float((B.sample(shape) * 2 - 1) * sparsity)

def local_binary_convolutional2d_layer(inputs, masks, filters, kernel_size=3, scope=None):
    in_channel = x.get_shape().as_list()[-1] # BHWC
    anchor_shape = [kernel_size, kernel_size, in_channel, masks]
    with tf.variable_scope(scope, 'local_binary_convolutional2d_layer', [inputs]):
        anchor = tf.get_variable('anchor', initializer=bernoulli(anchor_shape), trainable=False)
        kernel = tf.get_variable('kernel', shape=[1, 1, masks, filters])

        x = inputs
        x = tf.nn.conv2d(x, filter=anchor, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.relu(x)
        x = tf.nn.conv2d(x, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
        return x

def LBCNN(inputs, dropout_rate, is_training, scope=None):
    with tf.variable_scope(scope, 'LBCNN', [inputs]):
        x = inputs
        x = local_binary_convolutional2d_layer(x, 32, 32)
        x = tf.layers.max_pooling2d(x, 2, 2, padding='SAME')
        x = local_binary_convolutional2d_layer(x, 32, 64)
        x = tf.layers.max_pooling2d(x, 2, 2, padding='SAME')
        x = tf.layers.flatten(x)
        #x = tf.layers.dense(x, 1024, activation=tf.nn.relu)
        #x = tf.layers.dropout(x, rate=dropout_rate, training=is_training)
        x = tf.layers.dense(x, 10)
        return x
