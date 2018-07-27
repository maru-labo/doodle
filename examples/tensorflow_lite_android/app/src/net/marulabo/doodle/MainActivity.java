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
package net.marulabo.doodle;

import android.content.res.AssetFileDescriptor;
import android.app.ProgressDialog;
import android.graphics.Color;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import java.net.MalformedURLException;
import java.net.URL;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.MappedByteBuffer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.tensorflow.lite.Interpreter;

public class MainActivity extends AppCompatActivity {
  static final String TAG = "DOODLE";
  static final String MODEL_FILENAME = "doodle.tflite";
  static final int THREAD_NUM = 4;

  protected Interpreter tflite;
  protected Button recognize;
  protected Button clear;
  protected TextView result;
  protected CanvasView canvas;

  private float[] image = null;
  private float[] probabilities = null;

  private void print(String text){
    result.setText(result.getText() + text + "\n");
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    canvas = (CanvasView) findViewById(R.id.canvas);
    recognize = (Button) findViewById(R.id.recognize);
    result = (TextView) findViewById(R.id.result);
    recognize.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View v) {
        int pixels[] = canvas.getPixels();
        float image[] = new float[pixels.length];
        for(int i = 0; i < pixels.length; ++i){
          image[i] = Color.alpha(pixels[i]) / 255.f;
        }
        tflite.run(image, probabilities);

        result.setText("");
        int bestIndex = 0;
        float bestScore = probabilities[0];
        for(int i = 1; i < probabilities.length; ++i) {
          if(bestScore < probabilities[i]){
            bestScore = probabilities[i];
            bestIndex = i;
          }
        }
        print("予測: " + bestIndex);
        for(int i = 0; i < probabilities.length; ++i) {
          print(i + ": " + probabilities[i]);
        }
      }
    });
    clear = (Button) findViewById(R.id.clear);
    clear.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View v) {
        canvas.clear();
      }
    });

    // Initialize the model
    try {
      AssetFileDescriptor modelFd = getAssets().openFd(MODEL_FILENAME);
      MappedByteBuffer model = loadModelFile(modelFd);
      tflite = new Interpreter(model, THREAD_NUM);
      tflite.setUseNNAPI(true);
      recognize.setEnabled(true);
    } catch(IOException e) {
      print("モデルの読み込みに失敗しました。");
      return;
    }
  }

  private MappedByteBuffer loadModelFile(AssetFileDescriptor file) throws IOException {
    FileInputStream inputStream = new FileInputStream(file.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = file.getStartOffset();
    long declaredLength = file.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }
}
