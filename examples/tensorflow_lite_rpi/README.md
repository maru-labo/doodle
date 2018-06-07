
# Doodle model for TensorFlow Lite on Raspberry Pi

## How to build

You need to prepare `libtensorflow-lite.a`.
Refer to [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/g3doc/rpi.md).

Native compile:

```
$ cd ~
$ git clone https://github.com/tensorflow/tensorflow.git
$ git clone https://github.com/google/flatbuffers.git
$ gcc-6 main.cpp \
>   -I$HOME/tensorflow \
>   -I$HOME/flatbuffers/include \
>   --std=c++11 \
>   -L. -ltensorflow-lite \
>   -lstdc++ -lpthread -ldl -lm
```

Convert SavedModel to TFLite Model:

```
$ python convert.py
```

