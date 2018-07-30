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

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PixelFormat;
import android.graphics.PorterDuff;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.SurfaceHolder;
import android.view.SurfaceHolder.Callback;

public class CanvasView extends SurfaceView implements Callback {
  private SurfaceHolder holder;
  private Paint paint;
  private Path path;
  private Bitmap bitmap;
  private Canvas canvas;
  private float scale;
  final int VIEW_WIDTH = 28;
  final int VIEW_HEIGHT = 28;

  public CanvasView(Context context) {
    super(context);
    initialize();
  }

  public CanvasView(Context context, AttributeSet attrs) {
    super(context, attrs);
    initialize();
  }

  @Override
  public void surfaceCreated(SurfaceHolder holder) {
    float scaleX = getWidth() / VIEW_WIDTH;
    float scaleY = getHeight() / VIEW_HEIGHT;
    scale = scaleX > scaleY ? scaleY : scaleX;

    if (bitmap == null) {
      bitmap = Bitmap.createBitmap(VIEW_WIDTH, VIEW_HEIGHT, Bitmap.Config.ARGB_8888);
    }
    if (canvas == null) {
      canvas = new Canvas(bitmap);
    }
    clear();
  }

  @Override
  public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
  }

  @Override
  public void surfaceDestroyed(SurfaceHolder holder) {
    //bitmap.recycle();
  }

  void initialize(){
    super.setZOrderOnTop(true);
    holder = getHolder();
    holder.setFormat(PixelFormat.TRANSPARENT);
    holder.addCallback(this);

    paint = new Paint();
    paint.setColor(Color.BLACK);
    paint.setStyle(Paint.Style.STROKE);
    paint.setStrokeCap(Paint.Cap.ROUND);
    paint.setAntiAlias(true);
    paint.setStrokeWidth(1.5f);
  }

  public void clear() {
    canvas.drawColor(0, PorterDuff.Mode.CLEAR);
    Canvas canvas = holder.lockCanvas();
    canvas.drawColor(0, PorterDuff.Mode.CLEAR);
    holder.unlockCanvasAndPost(canvas);
  }

  public int[] getPixels() {
    int width = bitmap.getWidth();
    int height = bitmap.getHeight();
    int pixels[] = new int[width * height];
    bitmap.getPixels(pixels, 0, width, 0, 0, width, height);
    return pixels;
  }

  private void drawLine(Path path) {
    Canvas canvas = holder.lockCanvas();
    canvas.scale(scale, scale);
    canvas.drawColor(0, PorterDuff.Mode.CLEAR);
    canvas.drawBitmap(bitmap, 0, 0, null);
    canvas.drawPath(path, paint);
    holder.unlockCanvasAndPost(canvas);
  }

  @Override
  public boolean onTouchEvent(MotionEvent event) {
    float x = event.getX() / scale;
    float y = event.getY() / scale;
    switch (event.getAction()) {
      case MotionEvent.ACTION_DOWN:
        onTouchDown(x, y);
        break;

      case MotionEvent.ACTION_MOVE:
        onTouchMove(x, y);
        break;

      case MotionEvent.ACTION_UP:
        onTouchUp(x, y);
        break;

      default:
    }
    return true;
  }

  private void onTouchDown(float x, float y) {
    path = new Path();
    path.moveTo(x, y);
  }

  private void onTouchMove(float x, float y) {
    path.lineTo(x, y);
    drawLine(path);
  }

  private void onTouchUp(float x, float y) {
    path.lineTo(x, y);
    canvas.drawPath(path, paint);
    Canvas canvas = holder.lockCanvas();
    canvas.scale(scale, scale);
    canvas.drawColor(0, PorterDuff.Mode.CLEAR);
    canvas.drawBitmap(bitmap, 0, 0, null);
    holder.unlockCanvasAndPost(canvas);
  }
}
