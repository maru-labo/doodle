#!/bin/bash

docker run -it --rm \
    -v $(pwd)/export:/var/lib/tensorflow-model \
    -p 8500:8500 \
    ornew/tensorflow-serving-api-server \
    tensorflow_model_server \
    --port=8500 \
    --model_config_file=/var/lib/tensorflow-model/config.pb.txt \
    --enable_batching=true
