// 見やすさのためにエラーチェックを省略しています

#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <iostream>

// ヘッダファイルはGitHubのtensorflow/tensorflowリポジトリに含まれています
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"

auto const* model_filename = "./doodle.tflite";
char const* LABELS[10] = {
  "apple", "bed", "cat", "dog", "eye",
  "fish", "grass", "hand", "ice creame", "jacket",
};
int const IMAGE_HEIGHT  = 28;
int const IMAGE_WIDTH   = 28;
int const IMAGE_CHANNEL = 1;

int main() {
  // TFLiteモデルの読み込み
  std::unique_ptr<tflite::FlatBufferModel> model
      = tflite::FlatBufferModel::BuildFromFile(model_filename);

  // インタプリタを作成
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);
  interpreter->AllocateTensors();

  // inputに入力データを代入します。
  float* input = interpreter->typed_input_tensor<float>(0);
  // Nx28x28x1
  std::ifstream file("inputs.dat");
  if(file.fail()) {
    std::cerr << "Not found inputs.dat file." <<std::endl;
    return -1;
  }
  std::string data;
  while(file && std::getline(file, data)) {
    std::cout << "============================" << std::endl;
    for(int h = 0; h < IMAGE_HEIGHT; ++h) {
      for(int w = 0; w < IMAGE_WIDTH; ++w) {
        for(int c = 0; c < IMAGE_CHANNEL; ++c) {
          auto i = h * IMAGE_WIDTH * IMAGE_CHANNEL + w * IMAGE_CHANNEL + c;
          input[i] = static_cast<float>(data[i]) / 255.f;
          std::printf("%3.0f,", static_cast<float>(data[i]));
        }
      }
      std::cout << std::endl;
    }

    auto start = std::chrono::system_clock::now();

    // Invokeを実行すると、モデルの推論が実行されます。
    interpreter->Invoke();

    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>
      (end - start).count();
    std::printf("Inference time: %f(ms)\n", elapsed);
    // outputからモデルの推論結を取得可能です。
    float* output = interpreter->typed_output_tensor<float>(0);
    for(int i = 0; i < 10; ++i) {
      std::printf("%10s: %6.2f%%\n", LABELS[i], output[i] * 100.f);
    }
  }
}
