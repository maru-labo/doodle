#!/usr/bin/env bash

gcc-6 main.cpp \
  -I$HOME/tensorflow \
  -I$HOME/flatbuffers/include \
  --std=c++11 \
  -L. \
  -ltensorflow-lite \
  -lstdc++ -lpthread -ldl -lm \
  -o doodle

