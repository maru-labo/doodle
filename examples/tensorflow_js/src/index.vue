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


