# coding: utf-8

import os
import six
import tensorflow as tf

import metrics

def train_input_fn(training_dir, params):
    return _input_fn(training_dir, params, is_training=True)

def eval_input_fn(training_dir, params):
    return _input_fn(training_dir, params, is_training=False)

def _input_fn(data_dir, params, is_training):
    batch_size  = params.get('batch_size', 96)
    buffer_size = params.get('shuffle_buffer_size', 4096)
    cmp_type    = params.get('tfrecord_compression_type', 'GZIP')
    train_file  = params.get('train_tfrecord_file', 'train.tfr')
    test_file   = params.get('test_tfrecord_file', 'test.tfr')

    tfrecord = os.path.join(data_dir,
        train_file if is_training else test_file)

    return (tf.data.TFRecordDataset(tfrecord, compression_type=cmp_type)
        .map(_parse_example)
        .shuffle(buffer_size)
        .batch(batch_size)
        .repeat(-1 if is_training else 1)
        .make_one_shot_iterator()
        .get_next())

def _parse_example(example):
    features = tf.parse_single_example(example,  {
        'image': tf.FixedLenFeature([28, 28, 1], tf.float32),
        'label': tf.FixedLenFeature([]         , tf.int64),
    })
    label = features.pop('label')
    return features, label

def serving_input_fn(params):
    return tf.estimator.export.build_raw_serving_input_receiver_fn({
        'image': tf.placeholder(tf.float32, [None, 28, 28, 1], name='image')
    })()

def model_fn(features, labels, mode, params):
    num_classes   = params.get('num_classes', 10)
    batch_size    = params.get('batch_size', 96)
    learning_rate = params.get('learning_rate', 1e-4)
    init_stddev   = params.get('initializer_normal_stddev', 0.09)
    dropout_rate  = params.get('dropout_rate', 0.4)
    is_training   = mode == tf.estimator.ModeKeys.TRAIN

    image = features['image']
    assert image.get_shape().as_list() == [None, 28, 28, 1]

    initializer = tf.truncated_normal_initializer(stddev=init_stddev)

    with tf.variable_scope('model', initializer=initializer):
        x = image
        x = tf.layers.conv2d(x, 32, 5, padding='SAME', activation=tf.nn.relu)
        x = tf.layers.max_pooling2d(x, 2, 2, padding='SAME')
        x = tf.layers.conv2d(x, 64, 5, padding='SAME', activation=tf.nn.relu)
        x = tf.layers.max_pooling2d(x, 2, 2, padding='SAME')
        #x = tf.layers.flatten(x)
        #XXX: `tf.layers.flatten` contains unsupported op `StridedSlice` by TensorFlow.js. Use `tf.reshape` as bellow.
        x = tf.reshape(x, [-1, 7*7*64])
        x = tf.layers.dense(x, 1024, activation=tf.nn.relu)
        x = tf.layers.dropout(x, rate=dropout_rate, training=is_training)
        x = tf.layers.dense(x, 10)
        logits = x
        assert logits.get_shape().as_list() == [None, num_classes]

    predictions = {
        'probabilities': tf.nn.softmax(logits, name='probabilities'),
        'classes'      : tf.argmax(logits, axis=1, name='classes'),
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
            predictions=predictions,
            export_outputs={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    tf.estimator.export.PredictOutput(predictions),
            })

    with tf.variable_scope('losses'):
        cross_entropy_loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits)

        total_loss = tf.losses.get_total_loss()

    metric_ops = metrics.calculate(labels, predictions['classes'], num_classes)

    global_step = tf.train.get_or_create_global_step()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.variable_scope('optimizer'), tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        fit = optimizer.minimize(total_loss, global_step)

    tf.summary.image('image', image)
    tf.summary.scalar('total_loss', total_loss)
    summary_op = tf.summary.merge_all()

    if params.get('sagemaker_job_name', None) is not None:
        return tf.estimator.EstimatorSpec(mode=mode,
            loss=total_loss,
            train_op=fit,
            eval_metric_ops=metric_ops)

    # For local test.
    training_hooks = [
        tf.train.SummarySaverHook(
            save_steps=1,
            output_dir='./test/logs/doodle.train',
            summary_op=summary_op)
    ]
    evaluation_hooks = [
        tf.train.SummarySaverHook(
            save_steps=5,
            output_dir='./test/logs/doodle.eval',
            summary_op=summary_op)
    ]
    return tf.estimator.EstimatorSpec(mode=mode,
        loss=total_loss,
        train_op=fit,
        eval_metric_ops=metric_ops,
        evaluation_hooks=evaluation_hooks,
        training_hooks=training_hooks)
