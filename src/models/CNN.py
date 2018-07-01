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

import tensorflow as tf

def CNN(inputs, dropout_rate, is_training, scope=None):
    with tf.variable_scope(scope, 'CNN', [inputs]):
        x = inputs
        x = tf.layers.conv2d(x, 32, 3, padding='SAME', activation=tf.nn.relu)
        x = tf.layers.max_pooling2d(x, 2, 2, padding='SAME')
        x = tf.layers.conv2d(x, 64, 3, padding='SAME', activation=tf.nn.relu)
        x = tf.layers.max_pooling2d(x, 2, 2, padding='SAME')
        #x = tf.layers.flatten(x)
        #XXX: `tf.layers.flatten` contains unsupported op `StridedSlice` by TensorFlow.js. Use `tf.reshape` as bellow.
        x = tf.reshape(x, [-1, 7*7*64])
        #x = tf.layers.dense(x, 1024, activation=tf.nn.relu)
        #x = tf.layers.dropout(x, rate=dropout_rate, training=is_training)
        x = tf.layers.dense(x, 10)
        return x
