
# Doodle Recognition PWA with TensorFlow.js

This is an example [PWA](https://en.wikipedia.org/wiki/Progressive_Web_Apps)
that uses [TensorFlow.js](https://js.tensorflow.org/) and performs doodle recognition.

It loads a pre-trained CNN model that was converted to TensorFlow.js format
by [tfjs-converter](https://github.com/tensorflow/tfjs-converter).
The training code is shared at [the root of this repo](../../../..).

It runs on reasonably new Chrome, Safari, Firefox, Mobile Safari and Android Chrome
(not on Edge nor IE. Safari and Mobile Safari have a start up issue;
takes a long time until getting ready, but work OK once started).

You can try it out at: https://doodle-simple-test.netlify.com/

![Screenshot](https://i.imgur.com/G6g18ap.png)

### Building and Running

The following commands will start a web server on `localhost:8080`
and open a browser page with the demo.

```bash
cd tensorflow_js_simple
yarn  # Installs dependencies.
yarn start  # Starts a web server and opens a page. Also watches for changes.
```

After `$ yarn build`, the contents of `public` directory holds deployable files.
Note that those files need to be served via **https** to enable PWA features
(unless they are served from `localhost`).

### Notes on Debugging

This is a PWA, so the involved files will be cached
(it is different from the browser cache; it is Service Worker controlled cache).
Attention is required to make sure that each debug execution respects
the latest file changes. Please reference the document
["Debugging Service Workers"](https://developers.google.com/web/fundamentals/codelabs/debugging-service-workers/)
for details including the other aspects of debugging PWA and Service Workers.
