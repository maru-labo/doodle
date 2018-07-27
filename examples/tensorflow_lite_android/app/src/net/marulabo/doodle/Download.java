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
import android.os.AsyncTask;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;

public class Download extends AsyncTask<URL, Integer, Boolean> {
  private Context context;
  private String pathTo;

  Download(Context context, String pathTo) {
    this.context = context;
    this.pathTo = pathTo;
  }

  @Override
  protected Boolean doInBackground(URL... urls) {
    final byte[] buffer = new byte[4096];
    HttpURLConnection connection = null;
    InputStream input = null;
    OutputStream output = null;
    try {
      connection = (HttpURLConnection) urls[0].openConnection();
      connection.connect();

      int length = connection.getContentLength();

      input = connection.getInputStream();
      output = this.context.openFileOutput(pathTo, Context.MODE_PRIVATE);

      int totalBytes = 0;
      int bytes = 0;
      while ((bytes = input.read(buffer)) != -1) {
        output.write(buffer, 0, bytes);
        totalBytes += bytes;
        publishProgress((int)(totalBytes * 100.f / length));
      }
    } catch (IOException e) {
      e.printStackTrace();
    } finally {
      try {
        if(input != null){
          input.close();
        }
        if(output != null){
          output.close();
        }
      } catch (IOException e) {
        e.printStackTrace();
      }
      if (connection != null) {
        connection.disconnect();
      }
    }
    return false;
  }

  @Override
  protected void onPostExecute(Boolean success) {
    super.onPostExecute(success);
  }
}
