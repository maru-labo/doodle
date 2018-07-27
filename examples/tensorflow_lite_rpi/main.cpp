/*
Copyright (c) 2018 一般社団法人 MaruLabo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
 */
#include <vector>
#include <chrono>
#include <iostream>

// https://github.com/tensorflow/tensorflow
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"

#include "./util.hpp"

int const NUMBER_CLASSES = 10;
char const* LABELS[NUMBER_CLASSES] = {
  "apple", "bed", "cat", "dog", "eye",
  "fish", "grass", "hand", "ice creame", "jacket",
};
int const IMAGE_HEIGHT  = 28;
int const IMAGE_WIDTH   = 28;
int const IMAGE_CHANNEL = 1;
int const IMAGE_BYTES   = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNEL;

int main(int argc, char const* argv[]) {
  auto&& opt = parse_args(argc, argv);
  if(opt.model_path.empty()) {
    std::cerr << "Must specify the model file path." << std::endl;
    return -1;
  }

  TfLiteStatus status;
  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;

  // Loading TFlite model (.tflite)
  std::cout << "Loading model: " << opt.model_path << std::endl;
  model = tflite::FlatBufferModel::BuildFromFile(opt.model_path.c_str());
  if(!model) {
    std::cerr << "Failed to load the model." << std::endl;
    return -1;
  }
  std::cout << "The model was loaded successful." << std::endl;

  // Create TFLite interpreter.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  //interpreter->UseNNAPI(true); // for Android only

  // Show information and validation for model input/output.
  if(!is_valid_model(interpreter)) {
    return -1;
  }

  // Setup interpreter.
  interpreter->SetNumThreads(4);
  status = interpreter->AllocateTensors();
  if(is_error(status)) {
    std::cerr << "Failed to allocate the memory for tensors." << std::endl;
    return -1;
  }

  // Gettings mutable Input/Output Tensor.
  float* image         = interpreter->typed_input_tensor<float>(0);
  float* probabilities = interpreter->typed_output_tensor<float>(0);

  char data[IMAGE_BYTES];
  if(!load_data(opt.input_file, data, IMAGE_BYTES)) return -1;

  // Fill input tensor.
  for(int h = 0; h < IMAGE_HEIGHT; ++h) {
    for(int w = 0; w < IMAGE_WIDTH; ++w) {
      for(int c = 0; c < IMAGE_CHANNEL; ++c) {
        auto i = h * IMAGE_WIDTH * IMAGE_CHANNEL + w * IMAGE_CHANNEL + c;
        auto pixel = static_cast<float>(data[i]);
        assert(0.f <= pixel && pixel <= 255.f);
        image[i] = pixel / 255.f;

        // Show pixel for debug.
        static char const DEBUG_PIXEL[] = {' ', '+', '*'};
        int v = static_cast<int>(pixel / 100);
        std::cout << DEBUG_PIXEL[v] << ' ';
      }
    }
    std::cout << std::endl;
  }

  auto start = std::chrono::system_clock::now();

  status = interpreter->Invoke();

  auto end = std::chrono::system_clock::now();

  if(is_error(status)) {
    std::cerr << "Failed to invoke the interpreter." <<std::endl;
    return -1;
  }

  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>
    (end - start).count();
  std::printf("Inference time: %f(ms)\n", elapsed);

  // Show predictions.
  for(int i = 0; i < 10; ++i) {
    std::printf("%11s: %6.2f%%\n", LABELS[i], probabilities[i] * 100.f);
  }
}
