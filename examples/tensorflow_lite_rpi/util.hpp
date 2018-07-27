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
#pragma once
#include <string>
#include <sstream>
#include <fstream>

// https://github.com/tensorflow/tensorflow
#include "tensorflow/contrib/lite/interpreter.h"

bool is_error(TfLiteStatus const& status) {
  return status != kTfLiteOk;
}

template<typename Dims>
std::string get_shape_string(Dims const& dims) {
  std::ostringstream s;
  s << "[" << dims->data[0];
  for(int i = 1; i < dims->size; ++i) {
    s << ", " << dims->data[i];
  }
  s << "]";
  return s.str();
}

template<typename Tensor>
void print_tensor_info(Tensor const& tensor) {
  std::cout
      << "  Name  : " << tensor->name << std::endl
      << "  DType : " << tensor->type << std::endl
      << "  Shape : " << get_shape_string(tensor->dims) << std::endl
      << "  Bytes : " << tensor->bytes << std::endl;
}

bool is_valid_model(std::unique_ptr<tflite::Interpreter> const& interpreter) {
  int tensor_size = interpreter->tensors_size();
  auto&& inputs = interpreter->inputs();
  auto&& outputs = interpreter->outputs();
  std::cout << "Input tensors:" << std::endl;
  for(int i = 0; i < inputs.size(); ++i) {
    auto&& tensor = interpreter->tensor(inputs[i]);
    print_tensor_info(tensor);
  }
  std::cout << "Output tensors:" << std::endl;
  for(int i = 0; i < outputs.size(); ++i) {
    auto&& tensor = interpreter->tensor(outputs[i]);
    print_tensor_info(tensor);
  }
  if(inputs.size() != 1) {
    std::cerr << "Input tensors number must be 1. Got " << inputs.size() << std::endl;
    return false;
  }
  if(outputs.size() != 1) {
    std::cerr << "Output tensors number must be 1. Got " << outputs.size() << std::endl;
    return false;
  }
  return true;
}

int load_data(std::string const& filename, char buf[], int size) {
  if(filename.empty()) {
    // Load input data from stdin.
    std::cin.read(buf, size);
    if(std::cin.fail()) {
      std::cerr << "Failed to read input data from stdin." << std::endl;
      return -1;
    }
  } else {
    // Load input data from file.
    std::ifstream file(filename);
    file.read(buf, size);
    if(file.fail()) {
      std::cerr << "Not found input file: " << filename << std::endl;
      return -1;
    }
  }
}

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

