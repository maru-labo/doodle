
# Doodle for TensorFlow Lite on Raspberry Pi

![](./screenshot.png)

## How to use

Predict data given from standard input or specified file data.

The input data is a grayscale image of 28x28 pixels.
Each pixel is represented by 0 to 255. Therefore, it is binary data of 784 bytes.

Optional Arguments:

- `-f`: The binary data file path for input data.

## How to build

You need to prepare `libtensorflow-lite.a`.
Refer to [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/g3doc/rpi.md).

Native compile example:

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
$ python convert.py <path/to/saved_model/dir>
```

