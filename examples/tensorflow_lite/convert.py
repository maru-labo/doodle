# coding: utf-8

import tensorflow as tf
import tempfile
import subprocess
tf.contrib.lite.tempfile = tempfile
tf.contrib.lite.subprocess = subprocess

saved_model_dir = './export/Servo/1525815064/'
tflite_model_path = './doodle.tflite'

with tf.Graph().as_default() as graph, tf.Session(graph=graph) as sess:
    # SavedModelを読み込みます
    meta_graph = tf.saved_model.loader.load(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        saved_model_dir)

    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, sess.graph_def, ['classes','probabilities'])

    # 入力と出力のノードを取得します
    signature = meta_graph.signature_def['serving_default']
    image = graph.get_tensor_by_name(signature.inputs['image'].name)
    image.set_shape([1] + image.get_shape().as_list()[1:])
    print(image.get_shape())
    classes = graph.get_tensor_by_name(signature.outputs['classes'].name)
    probabilities = graph.get_tensor_by_name(signature.outputs['probabilities'].name)

    # FlatBuffersに変換してファイルに保存します
    tflite_model = tf.contrib.lite.toco_convert(
        output_graph_def, [image], [probabilities])
    open(tflite_model_path, 'wb').write(tflite_model)
