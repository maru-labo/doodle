<template lang='pug'>
  section
    canvas(width='28' height='28')
    .buttons.is-centered
      .button(@click='clear') Clear
      .button.is-primary(@click='recognize') Recognize
</template>

<style lang='stylus' scoped>
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
      fook: false,
      x: 0,
      y: 0,
      originX: 0,
      originY: 0,
    }),
    mounted(){
      this._canvas = this.$el.querySelector('canvas')
      this._canvas.addEventListener('touchstart', this.onDown, {passive: true})
      this._canvas.addEventListener('touchmove', this.onMove, {passive: true})
      this._canvas.addEventListener('touchend', this.onUp, {passive: true})

      this._canvas.addEventListener('mousedown', this.onMouseDown, false)
      this._canvas.addEventListener('mousemove', this.onMouseMove, false)
      this._canvas.addEventListener('mouseup', this.onMouseUp, false)
      this._context = this._canvas.getContext('2d')
      this._context.scale(0.1, 0.1)
      this._context.strokeStyle = 'black'
      this._context.lineWidth = 10
      this._context.lineJoin = 'round'
      this._context.lineCap = 'round'
      this.clear()
    },
    methods: {
      clear(){
        this._context.fillStyle = 'white'
        const {width, height} = this._canvas.getBoundingClientRect()
        this._context.fillRect(0, 0, width, height)
        this.$emit('clear')
      },
      recognize(){
        this.$emit('recognize', this.getImageData())
      },
      getImageData(){
        return this._context.getImageData(0, 0, 28, 28)
      },
      drawLine(){
        this._context.beginPath()
        this._context.moveTo(this.originX, this.originY)
        this._context.lineTo(this.x, this.y)
        this._context.stroke()
      },
      onDown(event){
        const {left, top} = event.target.getBoundingClientRect()
        const {pageX, pageY} = event.touches[0]
        this.originX = pageX - left
        this.originY = pageY - top
        this.fook = true
        event.stopPropagation()
      },
      onMove(event){
        if(this.fook){
          const {left, top} = event.target.getBoundingClientRect()
          const {pageX, pageY} = event.touches[0]
          this.x = pageX - left
          this.y = pageY - top
          this.drawLine()
          this.originX = this.x
          this.originY = this.y
          //event.preventDefault() // passive listener will be ignored this.
          event.stopPropagation()
        }
      },
      onUp(event){
        this.fook = false
        event.stopPropagation()
      },
      onMouseDown(event){
        const {left, top} = event.target.getBoundingClientRect()
        const {clientX, clientY} = event
        this.originX = clientX - left
        this.originY = clientY - top
      },
      onMouseMove(event){
        if(event.buttons == 1){ // Left clicking.
          const {left, top} = event.target.getBoundingClientRect()
          const {clientX, clientY} = event
          this.x = clientX - left
          this.y = clientY - top
          this.drawLine()
          this.originX = this.x
          this.originY = this.y
        }
      },
      onMouseUp(event){
      },
    },
  }
</script>
