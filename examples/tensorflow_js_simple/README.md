
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

To enable the PWA aspects (e.g. offline operations),
the contents of `public` directory (after `$ yarn build`)
needs to be served via **https**.
