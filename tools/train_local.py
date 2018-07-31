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
import tensorflow as tf
import tensorflow_hub as tfhub

def train_local(
    name,
    src_dir,
    data_dir,
    model_dir,
    run_config,
    train_spec,
    eval_spec,
    params):
  sys.path.append(src_dir)
  from doodle.inputs import train_input_fn, eval_input_fn, serving_input_fn
  from doodle.model import model_fn

  _model_dir = os.path.join(model_dir, name)
  _run_config = tf.estimator.RunConfig(
    model_dir=_model_dir,
    **run_config)

  _train_spec = tf.estimator.TrainSpec(
    input_fn=lambda: train_input_fn(data_dir, params),
    **train_spec)

  _eval_spec = tf.estimator.EvalSpec(
    input_fn=lambda: eval_input_fn(data_dir, params),
    exporters=[
      tf.estimator.LatestExporter('savedmodel', serving_input_fn(params)),
      tfhub.LatestModuleExporter('hub', serving_input_fn(params)),
    ],
    **eval_spec)

  estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    config=_run_config,
    params=params)

  tf.estimator.train_and_evaluate(estimator, _train_spec, _eval_spec)

  metrics = estimator.evaluate(_eval_spec.eval_input_fn, steps=_eval_spec.steps)
  print('###### metrics ' + '#' * 65)
  for name, value in sorted(six.iteritems(metrics)):
    print('{:<30}: {}'.format(name, value))

if __name__ == '__main__':
  import argparse
  import yaml
  p = argparse.ArgumentParser()
  p.add_argument('-c', '--config', default='config.yaml')
  p.add_argument('-L','--log-level', default='DEBUG')
  args = p.parse_args()
  config = yaml.load(open(args.config))
  if type(config) is list:
    for c in config:
      print(c)
      train_local(**c)
  else:
    train_local(**config)
