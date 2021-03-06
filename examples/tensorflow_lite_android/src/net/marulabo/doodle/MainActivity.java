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

import android.content.res.AssetManager;
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
  static final String MODEL_FILENAME = "downloaded";
  static final int THREAD_NUM = 4;
  static final int NUMBER_CLASSES = 10;
  static final int IMAGE_PIXELS = 784;
  static final int IMAGE_HEIGHT = 28;
  static final int IMAGE_WIDTH = 28;
  static final int IMAGE_CHANNEL = 1;
  static final int BATCH = 1;
  static final String[] LABELS = {
    "apple", "bed", "cat", "dog", "eye",
    "fish", "grass", "hand", "ice creame", "jacket",
  };

  protected Interpreter interpreter;
  protected Button recognize;
  protected Button clear;
  protected TextView result;
  protected CanvasView canvas;

  private float[][][][] image = null;
  private float[][] probabilities = null;

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
        // Fill input pixels.
        int pixels[] = canvas.getPixels();
        assert(pixels.length == IMAGE_PIXELS);
        for(int b = 0; b < BATCH; ++b){
          for(int h = 0; h < IMAGE_HEIGHT; ++h){
            for(int w = 0; w < IMAGE_WIDTH; ++w){
              for(int c = 0; c < IMAGE_CHANNEL; ++c){
                image[b][h][w][c] = Color.alpha(pixels[
                  c
                  + (IMAGE_CHANNEL * w)
                  + (IMAGE_CHANNEL * IMAGE_WIDTH * h)
                  + (IMAGE_CHANNEL * IMAGE_WIDTH * IMAGE_HEIGHT * b)
                ]) / 255.f;
              }
            }
          }
        }

        // Run inference.
        long startTime = System.nanoTime();
        interpreter.run(image, probabilities);
        long endTime = System.nanoTime();
        double ms = (endTime - startTime) / 1000000.0;

        // Show prediction result.
        result.setText("");
        print(String.format("Inference Time: %10f(ms)", ms));
        float[] prob = probabilities[0];
        int classes = argMax(prob);
        print(String.format("Predict: %s (%3.2f%%)",
              LABELS[classes], 100.f * prob[classes]));
        print("Details:");
        for(int i = 0; i < prob.length; ++i) {
          print(String.format("- %11s: %7.4f%%", LABELS[i], 100.f * prob[i]));
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

    // for Input/Output tensor.
    image = new float[BATCH][IMAGE_HEIGHT][IMAGE_WIDTH][IMAGE_CHANNEL];
    probabilities = new float[BATCH][NUMBER_CLASSES];

    // Initialize the model
    try {
      AssetManager assets = getAssets();
      MappedByteBuffer model = loadAssetToMemory(assets, MODEL_FILENAME);
      interpreter = new Interpreter(model, THREAD_NUM);
      interpreter.setUseNNAPI(true);
      recognize.setEnabled(true);
      print("The model was loaded successful.");
    } catch(IOException e) {
      print("Failed to load the model: " + e.getMessage() + "\n");
      return;
    }
  }

  private void print(String text){
    result.setText(result.getText() + text + "\n");
  }

  private int argMax(float[] array) {
    int index = 0;
    float max = array[0];
    for(int i = 1; i < array.length; ++i) {
      if(max < array[i]){
        max = array[i];
        index = i;
      }
    }
    return index;
  }

  private MappedByteBuffer loadAssetToMemory(AssetManager assets, String path) throws IOException {
    AssetFileDescriptor file = assets.openFd(path);
    FileInputStream stream = new FileInputStream(file.getFileDescriptor());
    FileChannel channel = stream.getChannel();
    long startOffset = file.getStartOffset();
    long declaredLength = file.getDeclaredLength();
    return channel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }
}
