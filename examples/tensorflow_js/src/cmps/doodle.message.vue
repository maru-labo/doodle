<template lang="pug">
  .comment-container
    template(v-if='result.label == "-"')
      | Please draw one of the classes.
    template(v-else-if='result.prob <= 0.3')
      p Sorry, I cannot understand...
    template(v-else)
      p.prob9(v-if='result.prob > 0.9')
        | This is "<b>{{result.label}}</b>"!!!
      p.prob7(v-else-if='result.prob > 0.7')
        | This is "{{result.label}}"!
      p.prob5(v-else-if='result.prob > 0.5')
        | This is "{{result.label}}".
      p.prob3(v-else-if='result.prob > 0.3')
        | This is "{{result.label}}"...?
      p {{result.prob*100}}%
</template>

<style lang="stylus" scoped>
  .prob0, .prob3, .prob5, .prob7, .prob9
    font-size: 2em
    padding: 0.3em
  .comment-container
    text-align: center
    height: 5em
</style>

<script>
  export default {
    props: {
      labels: {
        required: true,
      },
      predictions: {
        required: true,
      },
    },
    computed: {
      result() {
        if(!this.predictions) {
          return {label: '-', prob: 0}
        } else {
          const index = this.argmax(this.predictions)
          return {
            label: this.labels[index],
            prob: this.predictions[index],
          }
        }
      }
    },
    methods: {
      argmax(arr) {
        return arr.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1]
      }
    }
  }
</script>

