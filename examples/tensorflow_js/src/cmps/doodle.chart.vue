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
  .chart-container
    canvas
</template>

<style lang="stylus" scoped>
  .chart-container
      display : block
      width   : 100%
      height  : 380px
      padding : 0
      margin  : 1em auto
</style>

<script>
  import Chart from 'chart.js'
  import options from './chart.options.yml'
  export default {
    props: {
      labels: {
        required: true,
      },
      probabilities: {
        required: true,
      },
    },
    data: () => ({
      chart: null,
    }),
    watch: {
      probabilities(){
        const data = !this.probabilities ? [] : this.probabilities
        this.chart.data.datasets[0].data = data
        this.chart.update()
      },
    },
    mounted(){
      this._canvas = this.$el.querySelector('canvas')
      this.chart = new Chart(this._canvas, {
        type: 'bar',
        data: {
          labels: this.labels,
          datasets: [{
            label: 'Probabilities',
            data: this.probabilities,
          }]
        },
        options,
      })
    },
  }
</script>
