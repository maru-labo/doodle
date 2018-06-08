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
      this._canvas.addEventListener('touchstart', this.onDown, {passive: false, capture: true})
      this._canvas.addEventListener('touchmove', this.onMove, {passive: false, capture: true})
      this._canvas.addEventListener('touchend', this.onUp, {passive: false, capture: true})

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
        this.fook = true
        const {left, top} = event.target.getBoundingClientRect()
        const {clientX, clientY} = event.touches[0]
        this.originX = clientX - left
        this.originY = clientY - top
        event.stopPropagation()
      },
      onMove(event){
        if(this.fook){
          const {left, top} = event.target.getBoundingClientRect()
          const {clientX, clientY} = event.touches[0]
          this.x = clientX - left
          this.y = clientY - top
          this.drawLine()
          this.originX = this.x
          this.originY = this.y
          event.preventDefault() // passive listener will be ignored this.
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
