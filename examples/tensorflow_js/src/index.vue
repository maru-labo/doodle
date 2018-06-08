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
  div
    template(v-if='config != null')
      doodle(
        :model-url='config.modelUrl'
        :weights-url='config.weightsUrl'
        :labels='config.labels')
    template(v-else)
      p {{message}}
</template>

<script>
  import Doodle from './cmps/doodle.vue'
  export default {
    name: 'app-index',
    data: () => ({
      config: null,
      message: 'Now loading...',
    }),
    async created() {
      try {
        const res = await fetch('config.json')
        if(!res.ok) {
          const mes = await err.text()
          this.message = `Error: ${mes}`
          return
        }
        this.config = await res.json()
      }
      catch(err) {
        console.error(err)
        this.message = `Error: ${err}`
      }
    },
    components: {
      Doodle,
    }
  }
</script>


