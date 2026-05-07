# 使用 GSAP 动画库

## 一、概念说明

GSAP（GreenSock Animation Platform）是业界领先的 JavaScript 动画库，提供精确的时序控制、时间轴编排、缓动函数等功能。与 Vue 的 Transition 钩子结合使用时，可以实现极其复杂的动画效果。

```vue
<script setup>
import { ref } from 'vue'
import gsap from 'gsap'

const show = ref(true)
const boxRef = ref(null)

function onEnter(el, done) {
  gsap.fromTo(el,
    { opacity: 0, y: 50, scale: 0.8 },
    { opacity: 1, y: 0, scale: 1, duration: 0.6, ease: 'back.out(1.7)', onComplete: done }
  )
}

function onLeave(el, done) {
  gsap.to(el, {
    opacity: 0, y: -50, scale: 0.8,
    duration: 0.4, ease: 'power2.in', onComplete: done
  })
}
</script>

<template>
  <button @click="show = !show">切换</button>
  <Transition @enter="onEnter" @leave="onLeave">
    <div v-if="show" ref="boxRef" class="box">GSAP 动画</div>
  </Transition>
</template>
```

## 二、具体用法

### 2.1 基本用法

```js
gsap.to('.box', { x: 100, rotation: 360, duration: 1 })
gsap.from('.box', { opacity: 0, y: 50, duration: 0.5 })
gsap.fromTo('.box', { opacity: 0 }, { opacity: 1, duration: 1 })
```

### 2.2 时间轴（Timeline）

```js
const tl = gsap.timeline()
tl.to('.box1', { x: 100, duration: 0.5 })
  .to('.box2', { y: 100, duration: 0.5 }, '-=0.2') // 提前 0.2s 开始
  .to('.box3', { rotation: 180, duration: 0.5 })
```

### 2.3 缓动函数

```js
gsap.to('.box', { x: 200, ease: 'elastic.out(1, 0.3)' })
gsap.to('.box', { x: 200, ease: 'bounce.out' })
```

## 三、注意事项与常见陷阱

- GSAP 是付费库（核心功能免费），商业项目注意授权
- 在组件卸载时调用 `gsap.killTweensOf()` 清理动画，避免内存泄漏
- GSAP 操作 DOM 直接修改 style，与 Vue 的响应式系统无冲突但需注意时序
