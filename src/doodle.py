# coding: utf-8

import os
import tensorflow as tf

import metrics

def _input_fn(data_dir, params, is_training):
    #=========================================================
    # 学習/評価時の入力データを返します
    #
    # S3上のTFRecordファイルが`data_dir`にマウントされているので
    # 読み込んでシャッフルしたり前処理してデータを返します
    #=========================================================
    batch_size  = params.get('batch_size', 96)
    buffer_size = params.get('shuffle_buffer_size', 4096)
    cmp_type    = params.get('tfrecord_compression_type', 'GZIP')

    if is_training:
        tfrecord = params.get('train_tfrecord_file')
    else:
        tfrecord = params.get('test_tfrecord_file')
    tfrecord = os.path.join(data_dir, tfrecord)

    def _parse_record(record):
        features = tf.parse_single_example(record,  {
            'image': tf.FixedLenFeature([28, 28, 1], tf.float32),
            'label': tf.FixedLenFeature([]         , tf.int64),
        })
        label = features.pop('label')
        return features, label

    return (tf.data.TFRecordDataset(tfrecord, compression_type=cmp_type)
        .map(_parse_record)
        .shuffle(buffer_size)
        .batch(batch_size)
        .repeat(-1 if is_training else 1)
        .make_one_shot_iterator()
        .get_next())

def train_input_fn(training_dir, params):
    #=========================================================
    # 学習時の入力データを返します
    #=========================================================
    return _input_fn(training_dir, params, is_training=True)

def eval_input_fn(training_dir, params):
    #=========================================================
    # 評価時の入力データを返します
    #=========================================================
    return _input_fn(training_dir, params, is_training=False)

def serving_input_fn(params):
    #=========================================================
    # サービング時の入力形式を定義します
    #=========================================================
    return tf.estimator.export.build_raw_serving_input_receiver_fn({
        'image': tf.placeholder(tf.float32, [None, 28, 28, 1], name='image')
    })()

def model_fn(features, labels, mode, params):
    #=========================================================
    # ハイパーパラメータを取得します
    #=========================================================
    num_classes   = params.get('num_classes', 10)
    batch_size    = params.get('batch_size', 96)
    learning_rate = params.get('learning_rate', 1e-4)
    init_stddev   = params.get('initializer_normal_stddev', 0.09)
    dropout_rate  = params.get('dropout_rate', 0.4)
    is_training   = mode == tf.estimator.ModeKeys.TRAIN

    # 第一引数のfeaturesが入力データです
    image = features['image']
    assert image.get_shape().as_list() == [None, 28, 28, 1]

    # 変数の初期化関数です
    initializer = tf.truncated_normal_initializer(stddev=init_stddev)

    #=========================================================
    # モデルを定義します
    #=========================================================
    with tf.variable_scope('model', initializer=initializer):
        x = image
        x = tf.layers.conv2d(x, 32, 5, padding='SAME', activation=tf.nn.relu)
        x = tf.layers.max_pooling2d(x, 2, 2, padding='SAME')
        x = tf.layers.conv2d(x, 64, 5, padding='SAME', activation=tf.nn.relu)
        x = tf.layers.max_pooling2d(x, 2, 2, padding='SAME')
        x = tf.reshape(x, [-1, 7 * 7 * 64])
        x = tf.layers.dense(x, 1024, activation=tf.nn.relu)
        x = tf.layers.dropout(x, rate=dropout_rate, training=is_training)
        x = tf.layers.dense(x, 10)
        logits = x

        # 予測結果: クラスごとの離散確率分布、最も確率の高いクラスのインデクス
        predictions = {
            'probabilities': tf.nn.softmax(logits),
            'classes'      : tf.argmax(logits, axis=1),
        }

    # `mode`は実行モードです 推論(PREDICT)、学習(TRAIN)、評価(EVAL)モードがあります
    # 推論モードの場合は、以降の学習の計算を実行する必要はないので結果を返します
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
            predictions=predictions,
            export_outputs={
                'predictions': tf.estimator.export.PredictOutput(predictions),
            })

    #=========================================================
    # モデルの誤差を定義します
    #=========================================================
    with tf.variable_scope('losses'):
        # クロスエントロピーを計算して誤差に追加します
        cross_entropy_loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits)
        
        # モデルで追加された全ての誤差の総和を取得します
        total_loss = tf.losses.get_total_loss()

    #=========================================================
    # モデルを学習(=パラメータを最適化)します
    #=========================================================
    with tf.variable_scope('optimizer'):
        global_step = tf.train.get_or_create_global_step()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # total_loss(誤差の総和)が小さくなるように変数を更新します
            optimizer = tf.train.AdamOptimizer(learning_rate)
            fit = optimizer.minimize(total_loss, global_step)

    #=========================================================
    # 精度などの値を計算してサマリにまとめます
    #=========================================================
    with tf.variable_scope('metrics'):  
        metrics.add_summary(labels, predictions['classes'], num_classes)
    
    # 計算した値はサマリに追加することでログとしてS3に保存できます
    # ログはTensorBoardなどでグラフ化することができます
    tf.summary.image('image', image, family='inputs')
    tf.summary.scalar('total_loss', total_loss, family='losses')
    
    return tf.estimator.EstimatorSpec(mode=mode,
        loss=total_loss, train_op=fit)
