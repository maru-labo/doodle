'use strict';

// ref: https://github.com/akimach/tensorflow17-sampler/tree/master/tfjs-mnist
// ref: https://github.com/yukagil/tfjs-mnist-cnn-demo/

import * as tf from '@tensorflow/tfjs-core';
import {loadFrozenModel} from '@tensorflow/tfjs-converter';

const MODEL_FILENAME = 'saved_model_js/tensorflowjs_model.pb';
const WEIGHTS_FILENAME = 'saved_model_js/weights_manifest.json';

const IMAGE_WIDTH = 28;
const IMAGE_HEIGHT = 28;

let model;
let doodlePad;
let resultTable;
let uiManager;

window.onload = () => {
  // prepare URLs for saved model files converted for TF.js
  const href = window.location.href;
  const pathPrefix = href.substring(0, href.lastIndexOf('/') + 1);
  const modelUrl = pathPrefix + MODEL_FILENAME;
  const weightsUrl = pathPrefix + WEIGHTS_FILENAME;

  // load pre-trained model
  console.log('Model URL:', modelUrl);
  console.log('Weights URL:', weightsUrl);
  loadFrozenModel(modelUrl, weightsUrl).then(loadedModel => {
    console.log('Loaded model: ', loadedModel);
    model = loadedModel;

    blockUntilTFReady(); // workaround for mobile safari
    uiManager.init();
  });

  console.log('TensorFlow.js version:', tf.version);
  doodlePad = new DoodlePad('draw-area', window.reset);
  resultTable = new ResultTable('result-tbody');
  uiManager = new UIManager(doodlePad, resultTable)
}

window.recognize = () => {
  const inputImage = doodlePad.getImageData(IMAGE_WIDTH, IMAGE_HEIGHT);
  const scores = executeRecognition(inputImage);
  resultTable.update(scores);
  uiManager.update();
}

function executeRecognition(imageData) {
  // convert to tensor (shape: [width, height, channels])
  const grayscaled = tf.fromPixels(imageData, 1/* grayscale */).toFloat();
  // normalize
  const normalized = tf.div(grayscaled, tf.scalar(255));
  // reshape input format (shape: [batch_size, width, height, channels])
  const input = normalized.expandDims(0);

  // predict
  const results = model.execute({image_1: input});
  const predictions = results.dataSync();
  results.dispose();
  return predictions;
}

window.reset = () => {
  doodlePad.clear();
  resultTable.clear();
  uiManager.update();
}

class DoodlePad extends SignaturePad {
  constructor(canvasId, newSessionCb) {
    const canvas = document.getElementById(canvasId);
    super(canvas, {
      minWidth: 6,
      maxWidth: 6,
      penColor: 'white',
      backgroundColor: 'black',
    });
    this.canvas = canvas;
    this.tmpC2d_ = document.createElement('canvas').getContext('2d');
    this.newSession_ = true;
    this.off();
    this.onBegin = () => {
      if (this.newSession_) {
        this.newSession_ = false;
        this.clear();
        newSessionCb();
      }
    }
  }

  getImageData(outputWidth, outputHeight) {
    this.tmpC2d_.drawImage(this.canvas, 0, 0, outputWidth, outputHeight);
    const imageData = this.tmpC2d_.getImageData(0, 0, outputWidth, outputHeight);
    this.newSession_ = true;
    return imageData;
  }
}

class ResultTable {
  constructor(elemId) {
    this.domElement = document.getElementById(elemId);
    this.isClean = true;
    this.isPrevClean = true;
  }

  update(scores) {
    this.isPrevClean = this.isClean;
    this.isClean = false;
    const maxProbability = scores.indexOf(Math.max.apply(null, scores));
    console.assert(this.domElement.children.length === scores.length);
    for (let i = 0; i < scores.length; i++) {
      // -0 can be returned as a score on Safari, which breaks the chart
      const prob = (scores[i] >= 0) ? scores[i] : 0;
      const tr = this.domElement.children[i];
      const probTd = tr.children[1];
      const barTd = tr.children[2];
      const probText = scientificToDecimal(prob.toPrecision(6)).substring(0, 8);
      probTd.innerText = probText;
      barTd.firstChild.style = 'width: ' + (prob * 100) + '%';
      barTd.firstChild.innerHTML = '&nbsp;'; // need something to get displayed
      if (i === maxProbability) {
        tr.classList.add('is-selected');
      }
    }
  }

  clear() {
    this.isPrevClean = this.isClean;
    this.isClean = true;
    for (let tr of this.domElement.children) {
      tr.classList.remove('is-selected');
      const probTd = tr.children[1];
      const barTd = tr.children[2];
      probTd.innerText = '-';
      if (barTd.firstChild) {
        barTd.firstChild.style = 'width: 0%';
      }
    }
  }
}

class UIManager {
  constructor(doodlePad, resultTable) {
    this.doodlePad = doodlePad;
    this.resultTable = resultTable;
    this.header = document.getElementById('header');
    this.recognizeButton = document.getElementById('recognize-button');
    this.resetButton = document.getElementById('reset-button');
    this.outputColumn = document.getElementById('output-column');
  }

  init() {
    this.doodlePad.on();
    this.doodlePad.canvas.classList.remove('is-disabled');
    this.recognizeButton.classList.remove('is-loading');
    // 'with-anim-workaround' (an extra animation) is added to keep
    // the loading spinner running even after invoking blockUntilTFReady()
    // (bulma's spinner animation stops w/o this extra animation)
    this.recognizeButton.classList.remove('with-anim-workaround');
    this.update(false);
  }

  update(withScrollAnim = true) {
    this.doodlePad.canvas.classList.remove('is-disabled');
    this.recognizeButton.classList.remove('is-disabled');
    this.resetButton.classList.remove('is-danger');
    this.outputColumn.classList.remove('is-hidden-mobile-alt');
    iNoBounce.disable();

    const sh = document.documentElement.scrollHeight;
    const wh = window.innerHeight;
    const autoScroll = (this.isNarrow_() && (sh > wh)) || this.isOnMobile_();

    if (this.resultTable.isClean) {
      if (!this.resultTable.isPrevClean && autoScroll && withScrollAnim) {
        scrollIt(0, 300, 'easeOutQuad', () => { // scroll to the top
          this.outputColumn.classList.add('is-hidden-mobile-alt');
        });
      } else {
        this.outputColumn.classList.add('is-hidden-mobile-alt');
      }
      this.header.classList.remove('is-zero-height-mobile-alt');
      this.header.classList.add('is-max-height-mobile-alt');
      if (this.isOnMobile_()) {
        iNoBounce.enable();
      }
    } else {
      if (autoScroll) {
        if (withScrollAnim) {
          // scroll and pull the result table up
          scrollIt(this.resetButton, 300, 'easeOutQuad');
        }
        this.doodlePad.canvas.classList.add('is-disabled');
        this.resetButton.classList.add('is-danger');
      }
      this.recognizeButton.classList.add('is-disabled');
      this.header.classList.remove('is-max-height-mobile-alt');
      this.header.classList.add('is-zero-height-mobile-alt');
    }
  }

  isNarrow_() {
    // narrow layout if outputColumn comes below recognizeButton
    const pbY = this.recognizeButton.getBoundingClientRect().top;
    const ocY = this.outputColumn.getBoundingClientRect().top;
    return (pbY < ocY);
  }

  isOnMobile_() {
    // relies on the css settings
    const style = window.getComputedStyle(this.outputColumn);
    return (parseInt(style.marginTop) === 0);
  }
}

window.addEventListener('resize', event => {
  if (uiManager) {
    uiManager.update(false);
  }
});

// Workaround for mobile Safari (iOS 11.3 ＋ tfjs: "0.6.1")
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
