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
    Juefei-Xu, Felix, Vishnu Naresh Boddeti, and Marios Savvides. "Perturbative
    Neural Networks." Proceedings of the IEEE Conference on Computer Vision and
    Pattern Recognition (CVPR). Vol. 1. 2018.
"""

import tensorflow as tf

def noise(shape, level=1.0):
    N = tf.distributions.Uniform(low=-1.0, high=1.0)
    return N.sample(shape) * level

def perturbative_layer(inputs, masks, filters, level=1.0, scope=None):
    noise_shape = inputs.get_shape().as_list()[1:3] + [masks]
    with tf.variable_scope(scope, 'perturbative_layer', [inputs]):
        N = tf.get_variable('noise', initializer=noise(noise_shape, level), trainable=False)
        V = tf.get_variable('kernel', shape=[1, 1, masks, filters])

        x = inputs
        x = tf.nn.relu(tf.add(x, N))
        x = tf.nn.conv2d(x, filter=V, strides=[1, 1, 1, 1], padding='SAME')
        return x

def PNN(inputs, dropout_rate, is_training, scope=None):
    with tf.variable_scope(scope, 'PNN', [inputs]):
        x = inputs
        x = perturbative_layer(x, 32, 32)
        x = tf.layers.max_pooling2d(x, 2, 2, padding='SAME')
        x = perturbative_layer(x, 32, 64)
        x = tf.layers.max_pooling2d(x, 2, 2, padding='SAME')
        x = tf.layers.flatten(x)
        #x = tf.layers.dense(x, 1024, activation=tf.nn.relu)
        #x = tf.layers.dropout(x, rate=dropout_rate, training=is_training)
        x = tf.layers.dense(x, 10)
        print(x.shape)
        return x
