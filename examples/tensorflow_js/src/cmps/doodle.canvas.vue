<template lang="pug">
  section
    canvas
    .buttons.is-centered
      .button(@click="clear") Clear
      .button.is-primary(@click="recognize") Recognize
</template>

<style lang="stylus" scoped>
  canvas
    display block
    width 280px
    height 280px
    padding 0
    margin 1em auto
    border solid 1px
</style>

<script>
  export default {
    data: () => ({
      canvas: null,
      context: null,
      fook: false,
      x: 0,
      y: 0,
      originX: 0,
      originY: 0,
    }),
    mounted(){
      this.canvas = this.$el.querySelector('canvas')
      this.canvas.addEventListener("touchstart", this.onDown, false)
      this.canvas.addEventListener("touchmove", this.onMove, false)
      this.canvas.addEventListener("touchend", this.onUp, false)
      this.canvas.addEventListener("mousedown", this.onMouseDown, false)
      this.canvas.addEventListener("mousemove", this.onMouseMove, false)
      this.canvas.addEventListener("mouseup", this.onMouseUp, false)
      this.canvas.setAttribute('width', 28)
      this.canvas.setAttribute('height', 28)
      this.context = this.canvas.getContext('2d')
      this.context.scale(0.1, 0.1)
      this.context.strokeStyle="#000000"
      this.context.lineWidth = 10
      this.context.lineJoin  = "round"
      this.context.lineCap   = "round"
      this.clear()
    },
    methods: {
      clear(){
        this.context.fillStyle = 'rgb(255,255,255)'
        this.context.fillRect(0, 0,
          this.canvas.getBoundingClientRect().width,
          this.canvas.getBoundingClientRect().height)
        this.$emit('clear')
      },
      recognize(){
        this.$emit('recognize', this.getImageData())
      },
      getImageData(){
        return this.context.getImageData(0, 0, 28, 28)
      },
      drawLine(){
        this.context.beginPath()
        this.context.moveTo(this.originX, this.originY)
        this.context.lineTo(this.x, this.y)
        this.context.stroke()
      },
      onDown(event){
        this.fook = true
        this.originX = event.touches[0].pageX-event.target.getBoundingClientRect().left
        this.originY = event.touches[0].pageY-event.target.getBoundingClientRect().top
        event.stopPropagation()
      },
      onMove(event){
        if(this.fook){
          this.x = event.touches[0].pageX-event.target.getBoundingClientRect().left
          this.y = event.touches[0].pageY-event.target.getBoundingClientRect().top
          this.drawLine()
          this.originX = this.x
          this.originY = this.y
          event.preventDefault()
          event.stopPropagation()
        }
      },
      onUp(event){
        this.fook = false
        event.stopPropagation()
      },
      onMouseDown(event){
        this.originX = event.clientX-event.target.getBoundingClientRect().left
        this.originY = event.clientY-event.target.getBoundingClientRect().top
        this.fook = true
      },
      onMouseMove(event){
        if(this.fook){
          this.x = event.clientX-event.target.getBoundingClientRect().left
          this.y = event.clientY-event.target.getBoundingClientRect().top
          this.drawLine()
          this.originX = this.x
          this.originY = this.y
        }
      },
      onMouseUp(event){
        this.fook = false
      },
    },
  }
</script>
