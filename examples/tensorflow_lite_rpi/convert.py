# coding: utf-8

import tensorflow as tf

# Workaround for bugs. Details refer below:
# https://github.com/tensorflow/tensorflow/issues/15410
import tempfile
import subprocess
tf.contrib.lite.tempfile = tempfile
tf.contrib.lite.subprocess = subprocess

TAG = tf.saved_model.tag_constants.SERVING

def convert(saved_model_dir, tflite_model_path):
    with tf.Graph().as_default() as graph, tf.Session(graph=graph) as sess:
        meta_graph = tf.saved_model.loader.load(sess, [TAG], saved_model_dir)

        # Freeze variables.
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['classes','probabilities'])

        # Getting input/output tensor name from signature.
        sig = meta_graph.signature_def['serving_default']
        image = graph.get_tensor_by_name(sig.inputs['image'].name)
        image.set_shape([1] + image.get_shape().as_list()[1:])
        print(image.get_shape())
        # Not using `classes` because TFLite not support ArgMax.
        # classes = graph.get_tensor_by_name(signature.outputs['classes'].name)
        probabilities = graph.get_tensor_by_name(sig.outputs['probabilities'].name)

        # Convert FlatBuffers by TOCO.
        tflite_model = tf.contrib.lite.toco_convert(
            frozen_graph_def, [image], [probabilities])

        # Saved.
        open(tflite_model_path, 'wb').write(tflite_model)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('saved_model_dir')
    parser.add_argument('-o', '--output-path', default='./doodle.tflite')
    args = parser.parse_args()
    print(args)
    convert(args.saved_model_dir, args.output_path)

