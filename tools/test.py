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

import sys

import six
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity('DEBUG')

def train(src_dir, config):
    sys.path.append(src_dir)
    import doodle as dd

    params = {
        'train_tfrecord_file' : 'train.tfr',
        'test_tfrecord_file'  : 'test.tfr',
    }

    model_fn = dd.model_fn
    train_input_fn = lambda: dd.train_input_fn('./data', params)
    eval_input_fn  = lambda: dd.eval_input_fn('./data', params)

    e = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=config.get('model_dir', None),
        params=params)
    e.train(train_input_fn, max_steps=config.get('max_steps', 10))

    metrics = e.evaluate(eval_input_fn, steps=1)
    print('###### metrics ' + '#' * 65)
    for name, value in six.iteritems(metrics):
        print('{:<30}: {}'.format(name, value))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source-dir')
    parser.add_argument('-m', '--model-dir')
    parser.add_argument('--max-steps', type=int, default=10)
    args = parser.parse_args()
    config = dict(
        model_dir=args.model_dir,
        max_steps=args.max_steps,
    )
    train(args.source_dir, config)
