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
import sys

import six
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub

tf.logging.set_verbosity('DEBUG')

def local_train(src_dir, model_dir, max_steps, random_seed, model_type):
    sys.path.append(src_dir)
    import doodle

    model_dir = os.path.join(model_dir, model_type)
    config = tf.estimator.RunConfig(
        model_dir=model_dir,
        tf_random_seed=random_seed,
        save_summary_steps=10,
        log_step_count_steps=100,
        keep_checkpoint_max=5,
    )
    params = {
        'model_type': model_type,
        'num_classes'   : 10,
        'learning_rate' : 1e-4,
        'dropout_rate'  : 0.4,
        'train_batch_size' : 96,
        'test_batch_size'  : 96,
        'initializer_normal_stddev' : 0.09,
        'train_repeat_count' : -1,
        'tfrecord_compression_type' : 'GZIP',
        'tfrecord_parallel_reads_num' : None,
        'train_tfrecord_files' : ['train.tfr'],
        'test_tfrecord_files'  : ['test.tfr'],
        'shuffle_buffer_size'  : 1024,
        'prefetch_buffer_size' : 1024,
    }

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: doodle.inputs.train_input_fn('./data', params),
        max_steps=max_steps)

    eval_input_fn = lambda: doodle.inputs.eval_input_fn('./data', params)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        steps=10,
        start_delay_secs=10,
        throttle_secs=30,
        exporters=[
            tfhub.LatestModuleExporter('hub', doodle.inputs.serving_input_fn(params)),
        ])

    estimator = tf.estimator.Estimator(
        model_fn=doodle.estimator.model_fn,
        config=config,
        params=params)
    
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    metrics = estimator.evaluate(eval_input_fn, steps=10)
    print('###### metrics ' + '#' * 65)
    for name, value in sorted(six.iteritems(metrics)):
        print('{:<30}: {}'.format(name, value))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', type=str, choices=['CNN','LBCNN','PNN'])
    parser.add_argument('-s', '--source-dir')
    parser.add_argument('-m', '--model-dir')
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--random-seed', type=int, default=2018)
    args = parser.parse_args()
    local_train(
        src_dir=args.source_dir,
        model_dir=args.model_dir,
        max_steps=args.max_steps,
        random_seed=args.random_seed,
        model_type=args.model_type)
