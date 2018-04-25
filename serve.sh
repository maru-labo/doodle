#!/bin/bash

model_dir=/var/lib/tensorflow-model

docker run -it --rm \
    -v $(pwd)/model:$model_dir \
    -p 8500:8500 \
    ornew/tensorflow-serving-api-server \
    tensorflow_model_server \
    --port=8500 \
    --model_config_file=$model_dir/config.pb.txt \
    --enable_batching=true
