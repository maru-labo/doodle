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
      classes: {
        required: true,
      },
      probabilities: {
        required: true,
      },
    },
    computed: {
      result() {
        if(!this.probabilities) {
          return {label: '-', prob: 0}
        } else {
          return {
            label: this.labels[this.classes],
            prob: this.probabilities[this.classes],
          }
        }
      }
    },
  }
</script>

