# CSS 动画

## 一、概念说明

CSS 动画使用 `@keyframes` 定义关键帧，与 CSS 过渡的区别在于：动画可以定义**多个中间状态**，且在元素初次渲染时也能自动触发（过渡需要属性变化）。

在 `<Transition>` 中使用 CSS 动画时，将过渡属性定义在 `*-active` 类中即可。

```vue
<script setup>
import { ref } from 'vue'
const show = ref(true)
</script>

<template>
  <button @click="show = !show">切换</button>
  <Transition name="bounce">
    <div v-if="show" class="box">弹跳动画</div>
  </Transition>
</template>

<style>
.bounce-enter-active { animation: bounceIn 0.6s; }
.bounce-leave-active { animation: bounceOut 0.4s; }

@keyframes bounceIn {
  0% { transform: scale(0); opacity: 0; }
  50% { transform: scale(1.15); }
  100% { transform: scale(1); opacity: 1; }
}

@keyframes bounceOut {
  0% { transform: scale(1); opacity: 1; }
  100% { transform: scale(0); opacity: 0; }
}
</style>
```

## 二、具体用法

### 2.1 关键帧动画

使用 `@keyframes` 定义 0%~100% 的中间状态，实现复杂动画效果。

### 2.2 动画与过渡的区别

| 特性 | CSS Transition | CSS Animation |
|------|---------------|---------------|
| 中间状态 | 仅起止两个状态 | 可定义多个关键帧 |
| 初次渲染 | 不触发 | 自动触发 |
| 语法 | `transition: property duration` | `@keyframes` + `animation` |
| 循环 | 不支持 | 支持 `animation-iteration-count` |

### 2.3 同时使用过渡和动画

当 `*-active` 类同时包含 `transition` 和 `animation` 时，Vue 需要知道以哪个的持续时间为准（见 Ch16）。

## 四、常见动画效果

```css
/* 淡入 */
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

/* 从下方滑入 */
@keyframes slideUp {
  from { transform: translateY(30px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}

/* 旋转加载 */
@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

/* 脉冲效果 */
@keyframes pulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.05); }
}

/* 摇晃提示 */
@keyframes shake {
  0%, 100% { transform: translateX(0); }
  25% { transform: translateX(-5px); }
  75% { transform: translateX(5px); }
}
```

## 五、动画控制属性

```css
.box {
  animation-name: bounceIn;
  animation-duration: 0.6s;
  animation-timing-function: ease-out;
  animation-delay: 0s;
  animation-iteration-count: 1;    /* 或 infinite */
  animation-direction: normal;     /* 或 alternate */
  animation-fill-mode: both;       /* 动画前后保持首尾帧 */
}

/* 简写 */
.box {
  animation: bounceIn 0.6s ease-out both;
}
```

## 六、暂停与播放动画

```vue
<script setup>
import { ref } from 'vue'
const paused = ref(false)
</script>

<template>
  <div class="spinner" :style="{ animationPlayState: paused ? 'paused' : 'running' }"></div>
  <button @click="paused = !paused">{{ paused ? '播放' : '暂停' }}</button>
</template>

<style>
.spinner {
  animation: spin 1s linear infinite;
}
</style>
```

## 三、注意事项与常见陷阱

- `@keyframes` 名称不要与 CSS 类名冲突
- 动画结束由 `animationend` 事件决定，确保关键帧的 `100%` 状态与最终样式一致
- 使用 `animation-fill-mode: both` 可防止动画结束后的闪烁
- `animation-iteration-count: infinite` 会导致动画永不结束，Vue 无法自动移除类
- 动画性能：`transform` 和 `opacity` 的动画可由 GPU 加速，避免使用 `width`/`height`
