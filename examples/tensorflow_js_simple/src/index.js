'use strict';

import * as tf from '@tensorflow/tfjs-core';
import {loadFrozenModel} from '@tensorflow/tfjs-converter';

const MODEL_FILENAME = 'saved_model_js/tensorflowjs_model.pb';
const WEIGHTS_FILENAME = 'saved_model_js/weights_manifest.json';

let model;
let doodlePad;
let resultTable;

window.onload = () => {
  // prepare URLs for the saved model files converted for TF.js
  const href = window.location.href;
  const pathPrefix = href.substring(0, href.lastIndexOf('/') + 1);
  const modelUrl = pathPrefix + MODEL_FILENAME;
  const weightsUrl = pathPrefix + WEIGHTS_FILENAME;
  console.log('Model URL:', modelUrl);
  console.log('Weights URL:', weightsUrl);

  // load the pre-trained model
  loadFrozenModel(modelUrl, weightsUrl).then(loadedModel => {
    console.log('Loaded model:', loadedModel);
    model = loadedModel;

    blockUntilTFReady(); // workaround for mobile safari
    const recognizeButton = document.getElementById('recognize-button');
    recognizeButton.classList.remove('blinking');
    recognizeButton.innerHTML = 'Recognize';
  });

  doodlePad = new DoodlePad('draw-area');
  resultTable = new ResultTable('result-tbody');
}

window.recognize = () => {
  const inputImage = doodlePad.getImageData(28, 28);

  // convert to a tensor (shape: [width, height, channels])
  const grayscaled = tf.fromPixels(inputImage, 1/* grayscale */).toFloat();
  // normalize
  const normalized = tf.div(grayscaled, tf.scalar(255));
  // reshape the format (shape: [batch_size, width, height, channels])
  const input = normalized.expandDims(0);

  // perform recognition
  const results = model.execute({image_1: input});
  const scores = results.dataSync();
  results.dispose();

  resultTable.update(scores);
}

window.reset = () => {
  doodlePad.clear();
  resultTable.clear();
}

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
      } else {
        tr.classList.remove('highlight');
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

// Workaround for mobile Safari (iOS 11.3 ＋ tfjs-core 0.9.0)
// Even after loadFrozenModel()'s completion, model.execute() results in
// '[Error] WebGL: INVALID_ENUM: readPixels: invalid type'
// and blocks execution there for a while (and then resumes execution).
// Instead of blocking UI at the first doodle trial, the following function
// is invoked while initialization so that the loading spinner keep spinning
// until the system really becomes ready for accepting input.
function blockUntilTFReady() {
  let input = tf.zeros([1, 28, 28, 1]);
  const results = model.execute({image_1: input});
  const predictions = results.dataSync();
  results.dispose();
}

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('./service-worker.js')
    .then(success => console.log('Info: Service worker registered:', success))
    .catch(error => console.error('Error: Service worker not registered:', error));
} else {
  console.log('Warning: No support for Service Worker');
}
