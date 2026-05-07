# JavaScript 钩子

## 一、概念说明

Vue 的 `<Transition>` 组件提供了一系列 JavaScript 钩子事件，允许使用纯 JavaScript 控制动画的每个阶段。这在需要复杂逻辑控制、与第三方动画库集成时非常有用。

```vue
<script setup>
import { ref } from 'vue'
const show = ref(true)

function onBeforeEnter(el) {
  el.style.opacity = 0
  el.style.transform = 'translateY(30px)'
}

function onEnter(el, done) {
  el.offsetHeight // 强制 reflow
  el.style.transition = 'all 0.5s ease'
  el.style.opacity = 1
  el.style.transform = 'translateY(0)'
  el.addEventListener('transitionend', done)
}

function onLeave(el, done) {
  el.style.transition = 'all 0.3s ease'
  el.style.opacity = 0
  el.style.transform = 'translateY(30px)'
  el.addEventListener('transitionend', done)
}
</script>

<template>
  <button @click="show = !show">切换</button>
  <Transition
    @before-enter="onBeforeEnter"
    @enter="onEnter"
    @after-enter="() => console.log('进入完成')"
    @before-leave="(el) => el.style.opacity = 1"
    @leave="onLeave"
    @after-leave="() => console.log('离开完成')"
  >
    <div v-if="show">JavaScript 钩子控制</div>
  </Transition>
</template>
```

## 二、具体用法

### 2.1 钩子列表

| 钩子 | 触发时机 | 参数 |
|------|----------|------|
| `@before-enter` | 进入前 | `el` |
| `@enter` | 进入中 | `el, done` |
| `@after-enter` | 进入后 | `el` |
| `@enter-cancelled` | 进入被取消 | `el` |
| `@before-leave` | 离开前 | `el` |
| `@leave` | 离开中 | `el, done` |
| `@after-leave` | 离开后 | `el` |
| `@leave-cancelled` | 离开被取消 | `el` |

### 2.2 done 回调

`enter` 和 `leave` 钩子接收 `done` 回调，**必须调用**以通知 Vue 动画完成。

```js
function onEnter(el, done) {
  // 动画完成后调用 done
  setTimeout(done, 500)
}
```

### 2.3 CSS 与 JS 钩子同时使用

如果同时定义了 CSS 过渡类和 JS 钩子，JS 钩子中的 `done` 会等待 CSS 过渡结束。

## 三、注意事项与常见陷阱

- **必须调用 `done`**：不调用会导致 Vue 无限等待
- `el.offsetHeight` 强制浏览器 reflow，确保初始状态生效后再开始动画
- `@enter-cancelled` 只在 `v-if` 从 `true` 切回 `false` 时触发（进入被打断）
