
import * as tf from '@tensorflow/tfjs-core';
import {loadFrozenModel} from '@tensorflow/tfjs-converter';

const MODEL_FILENAME = 'saved_model_js/tensorflowjs_model.pb';
const WEIGHTS_FILENAME = 'saved_model_js/weights_manifest.json';
const INPUT_NODE_NAME = 'image_1';
const OUTPUT_NODE_NAME = 'probabilities';

document.addEventListener('DOMContentLoaded', async () => {
  // prepare URLs for the saved model files converted for TF.js
  const href = window.location.href;
  const pathPrefix = href.substring(0, href.lastIndexOf('/') + 1);
  const modelUrl = pathPrefix + MODEL_FILENAME;
  const weightsUrl = pathPrefix + WEIGHTS_FILENAME;
  console.log('Model URL:', modelUrl);
  console.log('Weights URL:', weightsUrl);

  // load the pre-trained model
  const model = await loadFrozenModel(modelUrl, weightsUrl);
  console.log('Loaded model:', model);

  blockUntilTFReady(); // workaround for mobile safari
  // make the UI ready for recognition
  const recognizeButton = document.getElementById('recognize-button');
  recognizeButton.classList.remove('blinking');
  recognizeButton.innerText = 'Recognize';

  const doodlePad = new DoodlePad('draw-area');
  const resultTable = new ResultTable('result-tbody');

  recognizeButton.addEventListener('click', () => {
    const inputImage = doodlePad.getImageData(28, 28);

    tf.tidy(() => {
      // convert the image to a tensor (shape: [width, height, channels])
      const grayscale = tf.fromPixels(inputImage, 1/* grayscale */).toFloat();
      // normalize the values
      const normalized = tf.div(grayscale, tf.scalar(255));
      // reshape the format (shape: [batch_size, width, height, channels])
      const input = normalized.expandDims(0);

      // perform recognition
      const output = model.execute({[INPUT_NODE_NAME]: input}, OUTPUT_NODE_NAME);
      const probabilities = output.dataSync();

      resultTable.update(probabilities);
    });
  });

  document.getElementById('clear-button').addEventListener('click', () => {
    doodlePad.clear();
    resultTable.clear();
  });

  // Workaround for mobile Safari (iOS 11.3 ＋ tfjs-core 0.9.0)
  // Even after loadFrozenModel()'s completion, model.execute() results in
  // '[Error] WebGL: INVALID_ENUM: readPixels: invalid type'
  // and blocks execution there for a while (and then resumes execution).
  // Instead of blocking UI at the first doodle trial, the following function
  // is invoked while initialization so that the UI keep showing loading status
  // until the system really becomes ready for accepting input.
  function blockUntilTFReady() {
    tf.tidy(() => model.execute({image_1: tf.zeros([1, 28, 28, 1])}));
  }
});

class DoodlePad extends SignaturePad {
  constructor(canvasId) {
    const canvas = document.getElementById(canvasId);
    super(canvas, {minWidth: 6, maxWidth: 6, penColor: 'white', backgroundColor: 'black'});
    this.canvas = canvas;
  }

  getImageData(width, height) {
    const c2d = document.createElement('canvas').getContext('2d');
    c2d.drawImage(this.canvas, 0, 0, width, height);
    const imageData = c2d.getImageData(0, 0, width, height);
    return imageData;
  }
}

class ResultTable {
  constructor(tbodyId) {
    this.domElement = document.getElementById(tbodyId);
  }

  update(scores) {
    const maxIndex = scores.indexOf(Math.max.apply(null, scores));
    for (let i = 0; i < scores.length; i++) {
      // -0 can be returned as a score on Safari, which breaks the chart
      const score = Math.max(scores[i], 0);
      const tr = this.domElement.children[i];
      tr.children[1].innerText = score.toFixed(5);
      tr.children[2].firstChild.style = 'width: ' + (score * 100) + '%';
      tr.children[2].firstChild.innerHTML = '&nbsp;'; // need something to get displayed
      if (i === maxIndex) {
        tr.classList.add('highlight');
      }
    }
  }

  clear() {
    for (let tr of this.domElement.children) {
      tr.classList.remove('highlight');
      tr.children[1].innerText = '-';
      tr.children[2].firstChild.style = 'width: 0%';
    }
  }
}

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('./service-worker.js')
    .then(success => console.log('Service Worker registered:', success))
    .catch(error => console.error('Error: Failed to register Service Worker:', error));
} else {
  console.log('Warning: No support for Service Workers');
}
