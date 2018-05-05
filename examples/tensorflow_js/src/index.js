'use strict'

import 'babel-polyfill'
import Vue from 'vue'
import Buefy from 'buefy'

Vue.use(Buefy)

import Index from './index.vue'

window.onload = () => {
  const pref = window.location.origin
  const app = new Vue({
    el: '#app',
    render: h => h(Index),
  })
}
