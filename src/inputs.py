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

import os
import tensorflow as tf
from util import with_default_params

def parse_example(example):
    features = tf.parse_single_example(example,  {
        'image': tf.FixedLenFeature([28, 28, 1], tf.float32),
        'label': tf.FixedLenFeature([]         , tf.int64),
    })
    label = features.pop('label')
    return features, label

def serving_input_fn(params):
    return tf.estimator.export.build_raw_serving_input_receiver_fn({
        'image': tf.placeholder(tf.float32, [None, 28, 28, 1], name='image')
    })

def _resolves(directories, filenames):
    return [os.path.join(directories, filename) for filename in filenames]

def train_input_fn(training_dir, params):
    params = with_default_params(params)
    data = tf.data.TFRecordDataset(
        _resolves(training_dir, params['train_tfrecord_files']),
        compression_type=params['tfrecord_compression_type'],
        num_parallel_reads=params['tfrecord_parallel_reads_num'])
    return (data
        .map(parse_example)
        .shuffle(params['shuffle_buffer_size'])
        .batch(params['train_batch_size'])
        .prefetch(params['prefetch_buffer_size'])
        .repeat(params['train_repeat_count'])
        .make_one_shot_iterator()
        .get_next())

def eval_input_fn(training_dir, params):
    params = with_default_params(params)
    data = tf.data.TFRecordDataset(
        _resolves(training_dir, params['test_tfrecord_files']),
        compression_type=params['tfrecord_compression_type'],
        num_parallel_reads=params['tfrecord_parallel_reads_num'])
    return (data
        .map(parse_example)
        .shuffle(params['shuffle_buffer_size'])
        .prefetch(params['prefetch_buffer_size'])
        .batch(params['test_batch_size'])
        .make_one_shot_iterator()
        .get_next())
