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

import tensorflow as tf
from tensorflowjs import quantization
from tensorflowjs.converters import tf_saved_model_conversion

DEFAULT_TAGS = [tf.saved_model.tag_constants.SERVING]
DEFAULT_SIGNATURE = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
DEFAULT_INPUTS = ['image']
DEFAULT_OUTPUTS = ['classes','probabilities']

def convert_to_webmodel(
    savedmodel_dir,
    output_dir,
    tags,
    signature,
    inputs,
    outputs,
    quantization_dtype,
    skip_op_check,
    strip_debug_ops
  ):
  with tf.Graph().as_default() as graph, tf.Session(graph=graph) as sess:
    meta_graph = tf.saved_model.loader.load(sess, tags, savedmodel_dir)
    meta = meta_graph.signature_def[signature]

    output_node_names = [
      meta.outputs[key].name for key in outputs
    ]

    # Getting input/output tensor name from signature for information.
    input_tensors = [
      graph.get_tensor_by_name(meta.inputs[key].name) for key in inputs
    ]
    output_tensors = [
      graph.get_tensor_by_name(name) for name in output_node_names
    ]
    print('input tensors:')
    for i, t in enumerate(input_tensors):
      print('  {}: "{}" {}'.format(i, t.name, t.shape.as_list()))
    print('output tensors:')
    for i, t in enumerate(output_tensors):
      print('  {}: "{}" {}'.format(i, t.name, t.shape.as_list()))

    tf_saved_model_conversion.convert_tf_saved_model(
      savedmodel_dir,
      output_node_names=','.join([n.split(':')[0] for n in output_node_names]),
      output_dir=output_dir,
      saved_model_tags=','.join(tags),
      quantization_dtype=quantization_dtype,
      skip_op_check=skip_op_check,
      strip_debug_ops=strip_debug_ops)

if __name__ == '__main__':
  import argparse
  p = argparse.ArgumentParser()
  p.add_argument('savedmodel_dir')
  p.add_argument('output_dir')
  p.add_argument('--tags', default=DEFAULT_TAGS)
  p.add_argument('--signature', default=DEFAULT_SIGNATURE)
  p.add_argument('--inputs', default=DEFAULT_INPUTS)
  p.add_argument('--outputs', default=DEFAULT_OUTPUTS)
  p.add_argument('--quantization_bytes', type=int, choices=set(quantization.QUANTIZATION_BYTES_TO_DTYPES.keys()))
  p.add_argument('--skip_op_check', default=False)
  p.add_argument('--strip_debug_ops', default=True)
  args = p.parse_args()

  quantization_dtype = (
    quantization.QUANTIZATION_BYTES_TO_DTYPES[args.quantization_bytes]
    if args.quantization_bytes else None)
  convert_to_webmodel(
    args.savedmodel_dir,
    args.output_dir,
    args.tags,
    args.signature,
    args.inputs,
    args.outputs,
    quantization_dtype,
    args.skip_op_check,
    args.strip_debug_ops
  )


