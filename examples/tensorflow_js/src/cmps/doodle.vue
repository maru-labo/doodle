<!--
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
-->

<template lang="pug">
  div.has-text-centered
    .hero.is-primary
      .hero-body
        .container
          h1.title.indie_flower
            | Doodle Recognition
          h2.subtitle with TensorFlow.js
    section.section
      .container.is-centerd
        .contents
          p Available Classes:
        .tags.is-centered
          span.tag.is-info.is-medium(v-for='label in labels') {{label}}
        doodle-canvas(@clear='clear' @recognize='recognize')
        doodle-message(:labels='labels' :probabilities='probabilities' :classes='classes')
        doodle-chart(:labels='labels' :probabilities='probabilities')
    footer.footer
      img(src='../img/marulabo.svg' width=96 height=96)
      p © 2018 MaruLabo All rights reserved.
</template>

<style lang="stylus">
  .indie_flower
    font-family: 'Indie Flower', cursive
</style>

<script>
  import * as tf from '@tensorflow/tfjs-core'
  import {loadFrozenModel} from '@tensorflow/tfjs-converter'

  import DoodleCanvas from './doodle.canvas.vue'
  import DoodleMessage from './doodle.message.vue'
  import DoodleChart from './doodle.chart.vue'
  export default {
    props: {
      modelUrl: {
        required: true,
      },
      weightsUrl: {
        required: true,
      },
      labels: {
        type: Array,
        required: true,
      },
    },
    data: () => ({
      probabilities: null,
      classes: null,
    }),
    async created() {
      console.log('Loading model from ', this.modelUrl, this.weightsUrl)
      try {
        this._model = await loadFrozenModel(this.modelUrl, this.weightsUrl)
      }
      catch(err) {
        console.error(err)
      }
      console.log('Loaded model: ', this._model)
    },
    methods: {
      clear() {
        this.probabilities = null
        this.classes = null
      },
      recognize(imageData) {
        tf.tidy(() => {
          const img = tf.fromPixels(imageData, 1).toFloat()
          const v255 = tf.scalar(255.)
          const grayscaled = tf.div(tf.sub(v255, img), v255)
          const results = this._model.execute({
            'image_1': grayscaled.expandDims(0)
          })
          this.probabilities = Array.from(results['probabilities'].dataSync())
          this.classes = Array.from(results['classes'].dataSync())
        })
      }
    },
    components: {
      DoodleCanvas,
      DoodleMessage,
      DoodleChart,
    }
  }
</script>

