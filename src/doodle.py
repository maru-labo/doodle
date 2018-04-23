# coding: utf-8

import os
import tensorflow as tf

import mobilenet.v2 as mobilenet_v2
import metrics

def _input_fn(data_dir, params, is_training):
    #=========================================================
    # 学習/評価時の入力データを返します
    #
    # S3上のTFRecordファイルが`data_dir`にマウントされているので
    # 読み込んでシャッフルしたり前処理してデータを返します
    #=========================================================
    batch_size  = params.get('batch_size', 512)
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
    num_classes  = params.get('num_classes', 10)
    initial_lr   = params.get('initial_learning_rate', 0.005)
    batch_size   = params.get('batch_size', 512)
    samples_num  = params.get('samples_per_epoch', 0)
    _decay_steps = int(samples_num * 2.5 / batch_size)
    decay_steps  = params.get('learning_rate_decay_steps', _decay_steps)
    decay_rate   = params.get('learning_rate_decay_rate', 0.94)
    init_stddev  = params.get('initializer_normal_stddev', 0.09)
    weight_decay = params.get('regularizer_weight_decay', 0.00004)
    keep_prob    = params.get('dropout_keep_prob', 0.999)
    is_training  = mode == tf.estimator.ModeKeys.TRAIN
    
    # 第一引数のfeaturesが入力データです
    image = features['image']
    assert image.get_shape().as_list() == [None, 28, 28, 1]
    
    # 変数の初期化と正則化の関数です
    initializer = tf.truncated_normal_initializer(stddev=init_stddev)
    regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
    
    #=========================================================
    # モデルを定義します
    #=========================================================
    with tf.variable_scope('model',
            initializer=initializer, regularizer=regularizer):
        
        # Google MobileNet v2 で入力データを変換します
        logits = mobilenet_v2.classify(image, num_classes, is_training, multiplier=0.25)
        
        # 出力をSoftmax関数で確率分布にします
        probabilities = tf.nn.softmax(logits, axis=1)
        
        # 確率の最も高いラベルのインデックスを返します
        classes = tf.argmax(probabilities, axis=1)
        
        # このモデルの出力は上記の確率分布と最も確率の高いラベルインデックスとします
        predictions = {
            'probabilities': probabilities,
            'classes': classes,
        }
    
    # `mode`は実行モードです 推論(PREDICT)、学習(TRAIN)、評価(EVAL)モードがあります
    # 推論モードの場合は、以降の学習の計算を実行する必要はないので、
    # 出力となる値を指定した`EstimatorSpec`を戻り値とします
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
        # ラベルをOne-Hot Vectorに変換します
        onehot_labels = tf.one_hot(labels, num_classes)
        
        # クロスエントロピーを計算して誤差に追加します
        cross_entropy_loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)
        # 正則化の損失を誤差に追加します
        regularization_loss = tf.losses.get_regularization_loss()
        # モデルで追加された全ての誤差の総和を取得します
        total_loss = tf.losses.get_total_loss()
    
    #=========================================================
    # モデルを学習(=パラメータを最適化)します
    #=========================================================
    with tf.variable_scope('optimizer'):
        # モデルの学習ステップ数を取得します
        global_step = tf.train.get_or_create_global_step()
        # 学習率を学習ステップ数などから決定します
        # ステップ数が進むにつれて、学習率が指数減衰するようにします
        learning_rate = tf.train.exponential_decay(
            initial_lr, global_step, decay_steps, decay_rate, staircase=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # ADAMという確率的最適化アルゴリズムを使います
            #   Kingma, Diederik P., and Jimmy Ba.
            #   "Adam: A method for stochastic optimization."
            #   arXiv preprint arXiv:1412.6980 (2014).
            optimizer = tf.train.AdamOptimizer(learning_rate)
            # total_loss(誤差の総和)が小さくなるように変数を更新します
            fit = optimizer.minimize(total_loss, global_step)

    #=========================================================
    # 精度などの値を計算してサマリにまとめます
    #=========================================================
    with tf.variable_scope('metrics'):  
        metrics.add_summary(labels, classes, num_classes)
    
    # 計算した値はサマリに追加することでログとしてS3に保存できます
    # ログはTensorBoardなどでグラフ化することができます
    tf.summary.image('image', image, family='inputs')
    tf.summary.scalar('cross_entropy_loss', cross_entropy_loss, family='losses')
    tf.summary.scalar('regularization_loss', regularization_loss, family='losses')
    tf.summary.scalar('total_loss', total_loss, family='losses')
    tf.summary.scalar('learning_rate', learning_rate, family='optimizer')
    
    return tf.estimator.EstimatorSpec(mode=mode,
        loss=total_loss, train_op=fit)
