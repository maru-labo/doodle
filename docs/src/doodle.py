# coding: utf-8

import os
import six
import tensorflow as tf

import metrics

def train_input_fn(training_dir, params):
    """
    学習時の入力データを返します
    """
    return _input_fn(training_dir, params, is_training=True)

def eval_input_fn(training_dir, params):
    """
    評価時の入力データを返します
    """
    return _input_fn(training_dir, params, is_training=False)

def _input_fn(data_dir, params, is_training):
    """
    学習/評価時の入力データを返します

    S3上のTFRecordファイルが`data_dir`にマウントされているので
    読み込んでシャッフルしたり前処理してデータを返します
    """
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
    """
    Example Protoをパースする関数です
    TFRecordの各レコードはExampleにエンコードしてあります
    """
    features = tf.parse_single_example(example,  {
        'image': tf.FixedLenFeature([28, 28, 1], tf.float32),
        'label': tf.FixedLenFeature([]         , tf.int64),
    })
    label = features.pop('label')
    return features, label

def serving_input_fn(params):
    """
    サービング時の入力形式を定義します
    """
    return tf.estimator.export.build_raw_serving_input_receiver_fn({
        'image': tf.placeholder(tf.float32, [None, 28, 28, 1], name='image')
    })()

def model_fn(features, labels, mode, params):
    """
    モデルを定義します
    """
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
    # ニューラルネットワークを定義します
    #=========================================================
    with tf.variable_scope('model', initializer=initializer):
        x = image
        x = tf.layers.conv2d(x, 32, 5, padding='SAME', activation=tf.nn.relu)
        x = tf.layers.max_pooling2d(x, 2, 2, padding='SAME')
        x = tf.layers.conv2d(x, 64, 5, padding='SAME', activation=tf.nn.relu)
        x = tf.layers.max_pooling2d(x, 2, 2, padding='SAME')
        x = tf.reshape(x, [-1, 7*7*64])
        x = tf.layers.dense(x, 1024, activation=tf.nn.relu)
        x = tf.layers.dropout(x, rate=dropout_rate, training=is_training)
        x = tf.layers.dense(x, 10)
        logits = x
        assert logits.get_shape().as_list() == [None, num_classes]

    # 予測結果: クラスごとの離散確率分布、最も確率の高いクラスのインデクス
    predictions = {
        'probabilities': tf.nn.softmax(logits, name='probabilities'),
        'classes'      : tf.argmax(logits, axis=1, name='classes'),
    }

    #=========================================================
    # 推論モードならモデルの出力を返します
    #=========================================================
    # `mode`は実行モードです。モードは以下の3つがあります。
    #     - tf.estimator.ModeKeys.PREDICT : 推論モードです
    #     - tf.estimator.ModeKeys.TRAIN   : 学習モードです
    #     - tf.estimator.ModeKeys.EVAL    : 評価モードです
    # 推論モードの場合は、学習を実行する必要はないため、結果を返して終わります。
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
            predictions=predictions,
            export_outputs={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    tf.estimator.export.PredictOutput(predictions),
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
    # 正答率など、モデルの評価値を計算します
    #=========================================================
    metric_ops = metrics.calculate(labels, predictions['classes'], num_classes)

    #=========================================================
    # モデルを学習(=パラメータを最適化)します
    #=========================================================
    global_step = tf.train.get_or_create_global_step()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.variable_scope('optimizer'), tf.control_dependencies(update_ops):
        # total_loss(誤差の総和)が小さくなるように変数を更新します
        optimizer = tf.train.AdamOptimizer(learning_rate)
        fit = optimizer.minimize(total_loss, global_step)

    #=========================================================
    # 変数をサマリにまとめます
    #=========================================================
    # 任意の値はサマリに追加することでログとして保存できます。
    # ログはTensorBoardなどでグラフ化することができるため、
    # 計算した誤差やメトリクス、入力画像などをログにしておきます。
    tf.summary.image('image', image)
    tf.summary.scalar('total_loss', total_loss)
    summary_op = tf.summary.merge_all()

    #=========================================================
    # これでモデルの定義は完了です！
    #=========================================================
    return tf.estimator.EstimatorSpec(mode=mode,
        loss=total_loss,
        train_op=fit,
        eval_metric_ops=metric_ops)
