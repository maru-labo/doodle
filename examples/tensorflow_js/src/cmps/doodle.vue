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
        doodle-message(:labels='labels' :predictions='predictions')
        doodle-chart(:labels='labels' :predictions='predictions')
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
      predictions: null,
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
        this.predictions = null
      },
      recognize(imageData) {
        const img = tf.fromPixels(imageData, 1).toFloat()
        const v255 = tf.scalar(255.)
        const grayscaled = tf.div(tf.sub(v255, img), v255)
        const results = this._model.execute({
          'image_1': grayscaled.expandDims(0)
        })
        const predictions = results.dataSync()
        results.dispose()
        this.predictions = Object.values(predictions)
      }
    },
    components: {
      DoodleCanvas,
      DoodleMessage,
      DoodleChart,
    }
  }
</script>

