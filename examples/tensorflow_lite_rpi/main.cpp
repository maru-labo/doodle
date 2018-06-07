/* Copyright (c) 2018 Arata Furukawa.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <iostream>

// https://github.com/tensorflow/tensorflow
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"

char const* LABELS[10] = {
  "apple", "bed", "cat", "dog", "eye",
  "fish", "grass", "hand", "ice creame", "jacket",
};
int const IMAGE_HEIGHT  = 28;
int const IMAGE_WIDTH   = 28;
int const IMAGE_CHANNEL = 1;
int const IMAGE_BYTES   = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNEL;

struct cli_options {
  bool binary_mode = true;
  std::string model_path;
  std::string input_file;
};

cli_options parse_args(int argc, char const* argv[]) {
  cli_options opt;
  int i = 1;
  while(i < argc) {
    if(std::strcmp("-f", argv[i]) == 0
    || std::strcmp("--file", argv[i]) == 0) {
      ++i;
      opt.input_file = argv[i];
      ++i;
    } else {
      opt.model_path = argv[i];
      ++i;
    }
  }
  return opt;
}

bool isError(TfLiteStatus const& status) {
  return status != kTfLiteOk;
}

int main(int argc, char const* argv[]) {
  auto&& opt = parse_args(argc, argv);
  if(opt.model_path.empty()) {
    std::cerr << "Must be specify model file path." << std::endl;
    return -1;
  }

  TfLiteStatus status;

  // Loading TFlite model (.tflite)
  std::cout << "Loading model: " << opt.model_path << std::endl;
  std::unique_ptr<tflite::FlatBufferModel> model
      = tflite::FlatBufferModel::BuildFromFile(opt.model_path.c_str());
  if(!model) {
    std::cerr << "Loading model failed." << std::endl;
    return -1;
  }
  std::cout << "The model was successfully loaded." << std::endl;

  // Create TFLite interpreter.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);
  status = interpreter->AllocateTensors();
  if(isError(status)) {
    std::cerr << "Failed to allocate tensor's memory." << std::endl;
    return -1;
  }

  // Show information for input/output.
  std::cout
    << "Input tensor numbers: " << interpreter->inputs().size() << std::endl
    << "Output tensor numbers: " << interpreter->outputs().size() << std::endl;

  // Mutable Input/Output Tensor.
  float* image         = interpreter->typed_input_tensor<float>(0);
  float* probabilities = interpreter->typed_output_tensor<float>(0);

  char data[IMAGE_BYTES];
  if(opt.input_file.empty()) {
    // Load input data from stdin.
    std::cin.read(data, IMAGE_BYTES);
    if(std::cin.fail()) {
      std::cerr << "Failed to read input data from stdin." << std::endl;
      return -1;
    }
  } else {
    // Load input data from file.
    std::ifstream file(opt.input_file);
    file.read(data, IMAGE_BYTES);
    if(file.fail()) {
      std::cerr << "Not found input file: " << opt.input_file << std::endl;
      return -1;
    }
  }

  // Fill input tensor.
  for(int h = 0; h < IMAGE_HEIGHT; ++h) {
    for(int w = 0; w < IMAGE_WIDTH; ++w) {
      for(int c = 0; c < IMAGE_CHANNEL; ++c) {
        auto i = h * IMAGE_WIDTH * IMAGE_CHANNEL + w * IMAGE_CHANNEL + c;
        auto pixel = static_cast<float>(data[i]);
        image[i] = pixel / 255.f;

        // Show pixel for debug.
        char const DEBUG_PIXEL[] = {' ', '+', '*'};
        int v = static_cast<int>(pixel / 100);
        std::cout << DEBUG_PIXEL[v] << ' ';
      }
    }
    std::cout << std::endl;
  }

  auto start = std::chrono::system_clock::now();

  // Run inference. Invoke output tensor.
  status = interpreter->Invoke();
  if(isError(status)) {
    std::cerr << "Failed the invocation of inference." <<std::endl;
    return -1;
  }

  auto end = std::chrono::system_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>
    (end - start).count();
  std::printf("Inference time: %f(ms)\n", elapsed);

  // Show predictions.
  for(int i = 0; i < 10; ++i) {
    std::printf("%11s: %6.2f%%\n", LABELS[i], probabilities[i] * 100.f);
  }
}
