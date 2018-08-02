#!/usr/bin/env bash

tensorflowjs_converter \
  --input_format=tf_saved_model \
  --output_node_names='probabilities,classes' \
  --saved_model_tags=serve \
  "$1" "$2"
