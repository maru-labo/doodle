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

import numpy as np

def merge(xs):
    y = dict()
    for x in xs:
        y.update(x)
    return y

DEFAULT_PARAMS = {
    'num_classes'   : 10,
    'train_batch_size' : 96,
    'test_batch_size'  : 256,
    'learning_rate' : 1e-4,
    'dropout_rate'  : 0.4,
    'initializer_normal_stddev' : 0.09,
    'train_repeat_count' : -1,
    'tfrecord_compression_type' : 'GZIP',
    'tfrecord_parallel_reads_num' : None,
    'train_tfrecord_files' : ['train.tfr'],
    'test_tfrecord_files'  : ['test.tfr'],
    'shuffle_buffer_size'  : 1024,
    'prefetch_buffer_size' : 1024,
}

def with_default_params(params):
    return merge([DEFAULT_PARAMS, params])

def metrics_scope_variables(scope):
    print('VariableScope: {}'.format(scope.name))
    gv = scope.global_variables()
    tv = scope.trainable_variables()
    gvp = int(np.sum([np.prod(i.shape) for i in gv]))
    tvp = int(np.sum([np.prod(i.shape) for i in tv]))
    print('  number of global    variables(parameters): {:,}({:,})'.format(len(gv), gvp))
    print('  number of trainable variables(parameters): {:,}({:,})'.format(len(tv), tvp))
