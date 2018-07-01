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
from models.CNN import CNN
from models.LBCNN import LBCNN
from models.PNN import PNN
from util import with_default_params, merge, metrics_scope_variables

def module_fn(params, is_training):
    image = tf.placeholder(tf.float32, [None, 28, 28, 1], name='image')
    initializer = tf.truncated_normal_initializer(stddev=params['initializer_normal_stddev'])

    with tf.variable_scope('model', initializer=initializer):
        if params['model_type'] == 'CNN':
            logits = CNN(image, params['dropout_rate'], is_training)
        elif params['model_type'] == 'LBCNN':
            logits = LBCNN(image, params['dropout_rate'], is_training)
        elif params['model_type'] == 'PNN':
            logits = PNN(image, params['dropout_rate'], is_training)
        else:
            raise 'unknown model_type: ' + params['model_type']
        #metrics_scope_variables(tf.get_variable_scope())

    with tf.variable_scope('predictions'):
        logits        = tf.identity(logits, name='logits')
        probabilities = tf.nn.softmax(logits, name='probabilities')
        classes       = tf.argmax(logits, axis=1, name='classes')
        assert logits.get_shape().as_list() == [None, params['num_classes']]

    tfhub.add_signature(
        name='default',
        inputs=image,
        outputs={
            'logits'       : logits,
            'probabilities': probabilities,
            'classes'      : classes,
        })

def model_fn(features, labels, mode, params):
    params = with_default_params(params)
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    spec = tfhub.create_module_spec(module_fn,
        tags_and_args=[
            ({'train'}, {'is_training': True, 'params': params}),
            (set(), {'is_training': False, 'params': params}),
        ])

    tag = {'train'} if is_training else None
    module = tfhub.Module(spec, trainable=is_training, tags=tag)
    tfhub.register_module_for_export(module, 'doodle')

    image = features['image']
    assert image.get_shape().as_list() == [None, 28, 28, 1]
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

    global_step = tf.train.get_or_create_global_step()

    metric_ops = metrics.calculate(labels, predictions['classes'], params['num_classes'])
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode,
            loss=total_loss,
            eval_metric_ops=metric_ops)

    with tf.variable_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(params['learning_rate'])
        with tf.control_dependencies(update_ops):
            fit = optimizer.minimize(total_loss, global_step)

    return tf.estimator.EstimatorSpec(mode=mode,
        loss=total_loss,
        train_op=fit,
        eval_metric_ops=metric_ops)
