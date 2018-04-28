'use strict'

import 'babel-polyfill'
import Vue from 'vue'
import Buefy from 'buefy'

Vue.use(Buefy)

import Doodle from './cmps/doodle.vue'

window.onload = () => {
  const pref = window.location.origin
  const app = new Vue({
    el: '#app',
    render: h => h(Doodle, {
      props: {
        modelUrl: `${pref}/saved_model_js/tensorflowjs_model.pb`,
        weightsUrl: `${pref}/saved_model_js/weights_manifest.json`,
        labels: [
          'apple','bed','cat','dog','eye',
          'fish','grass','hand','ice creame','jacket'
        ],
      }
    })
  })
}
