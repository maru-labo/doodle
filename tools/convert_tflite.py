# coding: utf-8
# Copyright (c) 2018 Arata Furukawa
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

# Workaround for bugs. Details refer below:
# https://github.com/tensorflow/tensorflow/issues/15410
import tempfile
import subprocess
tf.contrib.lite.tempfile = tempfile
tf.contrib.lite.subprocess = subprocess

DEFAULT_TAGS = [tf.saved_model.tag_constants.SERVING]
DEFAULT_SIGNATURE = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
DEFAULT_INPUTS = ['image']
# Not using `classes` because TFLite not support ArgMax.
DEFAULT_OUTPUTS = ['probabilities']

def convert_to_tflite(
    savedmodel_dir,
    output_path,
    tags,
    signature,
    inputs,
    outputs,
    quantized=True,
  ):
  with tf.Graph().as_default() as graph, tf.Session(graph=graph) as sess:
    meta_graph = tf.saved_model.loader.load(sess, tags, savedmodel_dir)
    meta = meta_graph.signature_def[signature]

    # Freeze variables.
    output_node_names = [
      meta.outputs[key].name for key in outputs
    ]
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
      sess, sess.graph_def, [
        n.split(':')[0] for n in output_node_names
      ])

    def fix_shape(t):
      if not t.shape.is_fully_defined():
        t.set_shape([
          d if d is not None else 1 for d in t.shape.as_list()
        ])
      return t

    # Getting input/output tensor name from signature.
    input_tensors = map(fix_shape, [
      graph.get_tensor_by_name(meta.inputs[key].name) for key in inputs
    ])
    output_tensors = [
      graph.get_tensor_by_name(name) for name in output_node_names
    ]

    print('input tensors:')
    for i, t in enumerate(input_tensors):
      print('  {}: "{}" {}'.format(i, t.name, t.shape.as_list()))
    print('output tensors:')
    for i, t in enumerate(output_tensors):
      print('  {}: "{}" {}'.format(i, t.name, t.shape.as_list()))

    # Convert to FlatBuffers by TOCO.
    tflite_model = tf.contrib.lite.toco_convert(
      frozen_graph_def, input_tensors, output_tensors,
      quantize_weights=quantized)

    # Save
    with open(output_path, 'wb') as f:
      f.write(tflite_model)

if __name__ == '__main__':
  import argparse
  p = argparse.ArgumentParser()
  p.add_argument('savedmodel_dir')
  p.add_argument('output_path')
  p.add_argument('-t','--tags', default=DEFAULT_TAGS)
  p.add_argument('-s','--signature', default=DEFAULT_SIGNATURE)
  p.add_argument('-i','--inputs', default=DEFAULT_INPUTS)
  p.add_argument('-o','--outputs', default=DEFAULT_OUTPUTS)
  p.add_argument('-q', '--quantized', action='store_true', default=False)
  args = p.parse_args()
  print('options:')
  for a, v in sorted(six.iteritems(vars(args))):
    print('  {}: {}'.format(a,v))
  convert_to_tflite(
    args.savedmodel_dir,
    args.output_path,
    args.tags,
    args.signature,
    args.inputs,
    args.outputs,
    args.quantized,
  )

