<template lang="pug">
  .chart-container
    canvas
</template>

<style lang="stylus" scoped>
  .chart-container
      display : block
      width   : 420px
      height  : 380px
      padding : 0
      margin  : 1em auto
</style>

<script>
  import Chart from 'chart.js'
  export default {
    props: {
      labels: {
        required: true,
      },
      predictions: {
        required: true,
      },
    },
    data: () => ({
      chart: null,
    }),
    watch: {
      predictions(){
        const data = !this.predictions ? [] : this.predictions
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
            data: this.predictions,
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            yAxes: [{
              ticks: {
                max: 1,
                min: 0,
                stepSize: 0.1,
              }
            }]
          }
        }
      })
    },
  }
</script>
