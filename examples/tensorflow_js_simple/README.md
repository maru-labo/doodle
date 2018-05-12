
# Doodle Recognition PWA with TensorFlow.js

This is an example [PWA](https://en.wikipedia.org/wiki/Progressive_Web_Apps)
that uses [TensorFlow.js](https://js.tensorflow.org/) and performs doodle recognition.

It loads a pre-trained CNN model that was converted to TensorFlow.js format
by [tfjs-converter](https://github.com/tensorflow/tfjs-converter).
The training code is shared at [the root of this repo](../../../..).

<img src="https://i.imgur.com/XOQIx5W.png" width="100" align="right">

It runs on reasonably new Chrome, Safari, Firefox, Mobile Safari and Android Chrome
(not on Edge nor IE. Safari and Mobile Safari have a start up issue;
takes a long time until getting ready, but work OK once started).

You can try it out at: https://tfjs-doodle-recognition-pwa.netlify.com/

![Screenshot](https://i.imgur.com/G6g18ap.png)

### Building and Running

[`tensorflowjs_converter`](https://github.com/tensorflow/tfjs-converter)
needs to be installed.<br/>
The following commands will start a web server on `localhost:8080`
and open a browser page with the demo.
The build process installs the pre-trained model of
[Release v1.0.0](https://github.com/maru-labo/doodle/releases/tag/v1.0.0).

```bash
cd tensorflow_js_simple
yarn        # Installs dependencies.
yarn start  # Starts a web server and opens a page. Also watches for changes.
```

After `yarn build`, `public` directory holds the deployable files.
Note that those files need to be served via **https** to enable PWA features
(unless they are served from `localhost`).

### <a name="notes-on-debugging"></a>Notes on Debugging

This is a PWA, so the involved files will be cached
(it is different from the browser cache; it is Service Worker controlled cache).
Attention is required to make sure that each debug execution respects
the latest file changes. Please reference the document
["Debugging Service Workers"](https://developers.google.com/web/fundamentals/codelabs/debugging-service-workers/)
for details including the other aspects of debugging PWA and Service Workers.

### Updating Pre-trained Model Data

The following steps illustrate how to update the model files.
You may wan to update `prepare_model` script for a new model.

1. Obtain saved model files of newly trained model.
2. Find out model information, necessary for conversion to TensorFlow.js format
    and subsequent execution with TensorFlow.js, by using
    [SavedModel CLI (Command-Line Interface)](https://www.tensorflow.org/versions/r1.2/programmers_guide/saved_model_cli)
3. Convert the saved model into TensorFlow.js format by using
    [tfjs-converter](https://github.com/tensorflow/tfjs-converter).
4. Replace the model files under `public/saved_model_js` with the updated ones.
5. Update the source code accordingly, if required.

[SavedModel CLI](https://www.tensorflow.org/versions/r1.2/programmers_guide/saved_model_cli)
and
[tfjs-converter](https://github.com/tensorflow/tfjs-converter)
need to be installed to move forward.

```bash
# This is an example; update the command lines as needed.
# Step 1
tar -xzf model.tar.gz
# This will yield a directory ('export') that contains the model files.

# Step 2
saved_model_cli show --dir export/Servo/* --all
# This will emits lines like the followings:
#     MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:
#
#     signature_def['serving_default']:
#       The given SavedModel SignatureDef contains the following input(s):
#         inputs['image'] tensor_info:
#             dtype: DT_FLOAT
#             shape: (-1, 28, 28, 1)
#             name: image_1:0
#       The given SavedModel SignatureDef contains the following output(s):
#         outputs['classes'] tensor_info:
#             dtype: DT_INT64
#             shape: (-1)
#             name: classes:0
#         outputs['probabilities'] tensor_info:
#             dtype: DT_FLOAT
#             shape: (-1, 10)
#             name: probabilities:0
#       Method name is: tensorflow/serving/predict

# Step 3, 4
# Read names of the model tag and the output nodes from the above output and run the coveter:
tensorflowjs_converter \
    --input_format=tf_saved_model \
    --saved_model_tags='serve' \
    --output_node_names='classes,probabilities' \
    export/Servo/* \
    public/saved_model_js

# Step 5
# Update INPUT_NODE_NAME and OUTPUT_NODE_NAME in src/index.js as needed.
```
Note that, for simplicity's sake, the example code deals with single output node.
This limitation has no drawback in this example
as `probabilities` node holds all the information necessary for the demo.
When you need to read output from more than one node, however, please reference
[the other example](https://github.com/maru-labo/doodle/blob/62c71ba554f827d77e907f517e2d585165cfc58b/examples/tensorflow_js/src/cmps/doodle.vue#L72-L76).

Another note -- as mentioned in ["Notes on Debugging"](#notes-on-debugging) above,
attention needs to be paid to make sure that the new files are really being used
when running after an update.
