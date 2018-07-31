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

import six
import tensorflow as tf
import tensorflow_hub as tfhub

import metrics

def module_fn(params, training):
  image = tf.placeholder(tf.float32, [None, 28, 28, 1], name='image')
  initializer = tf.glorot_uniform_initializer()

  with tf.variable_scope('model', initializer=initializer):
    x = image
    x = tf.layers.conv2d(x, 32, 3, padding='SAME', activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(x, 2, 2, padding='SAME')
    x = tf.layers.conv2d(x, 64, 3, padding='SAME', activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(x, 2, 2, padding='SAME')
    #XXX: `tf.layers.flatten` contains unsupported op `StridedSlice` by
    #     TensorFlow.js. Use `tf.reshape` as bellow.
    #x = tf.layers.flatten(x)
    x = tf.reshape(x, [-1, 7*7*64])
    x = tf.layers.dense(x, 1024, activation=tf.nn.relu)
    x = tf.layers.dropout(x, rate=params['dropout_rate'], training=training)
    x = tf.layers.dense(x, params['num_classes'])
    x.shape.assert_is_compatible_with([None, params['num_classes']])

  with tf.variable_scope('predictions'):
    logits        = tf.identity(x, name='logits')
    probabilities = tf.nn.softmax(logits, name='probabilities')
    classes       = tf.argmax(logits, axis=1, name='classes')

  tfhub.add_signature(
    name='default',
    inputs=image,
    outputs={
      'logits'       : logits,
      'probabilities': probabilities,
      'classes'      : classes,
    })

def model_fn(features, labels, mode, params):
  training = mode == tf.estimator.ModeKeys.TRAIN

  spec = tfhub.create_module_spec(module_fn,
    tags_and_args=[
      ({'train'}, {'training': True,  'params': params}),
      (set(),     {'training': False, 'params': params}),
    ])

  tags = {'train'} if training else None
  module = tfhub.Module(spec, trainable=training, tags=tags)
  tfhub.register_module_for_export(module, 'doodle')

  image = features['image']
  image.shape.assert_is_compatible_with([None, 28, 28, 1])
  predictions = module(image, as_dict=True)

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode,
      predictions=predictions,
      export_outputs={
        tf.saved_model.signature_constants.
        DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            tf.estimator.export.PredictOutput(predictions),
      })

  with tf.variable_scope('losses'):
    cross_entropy_loss = tf.losses.sparse_softmax_cross_entropy(
      labels=labels, logits=predictions['logits'])
    total_loss = tf.losses.get_total_loss()

  tf.summary.image('image', image)
  tf.summary.scalar('total_loss', total_loss)

  metric_ops = metrics.calculate(
    labels, predictions['classes'], params['num_classes'])

  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(mode=mode,
      loss=total_loss,
      eval_metric_ops=metric_ops)

  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_or_create_global_step()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.variable_scope('optimizer'):
      optimizer = tf.train.AdamOptimizer(params['learning_rate'])
      with tf.control_dependencies(update_ops):
        fit = optimizer.minimize(total_loss, global_step)
    return tf.estimator.EstimatorSpec(mode=mode,
      loss=total_loss,
      train_op=fit,
      eval_metric_ops=metric_ops)
