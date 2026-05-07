# CSS 过渡类名

## 一、概念说明

Vue 在 `<Transition>` 中会自动为元素应用 6 个 CSS 类名，分为**进入**和**离开**两组。这些类名控制了动画的起始状态、激活阶段和结束状态。

进入过程：`enter-from` → `enter-active` → `enter-to`
离开过程：`leave-from` → `leave-active` → `leave-to`

```vue
<script setup>
import { ref } from 'vue'
const show = ref(true)
</script>

<template>
  <button @click="show = !show">切换</button>
  <Transition name="slide">
    <div v-if="show" class="panel">滑动面板</div>
  </Transition>
</template>

<style>
/* 进入动画 */
.slide-enter-from { transform: translateX(-100%); opacity: 0; }
.slide-enter-active { transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1); }
.slide-enter-to { transform: translateX(0); opacity: 1; }

/* 离开动画 */
.slide-leave-from { transform: translateX(0); opacity: 1; }
.slide-leave-active { transition: all 0.3s ease-in; }
.slide-leave-to { transform: translateX(100%); opacity: 0; }
</style>
```

## 二、具体用法

### 2.1 六个类名详解

| 类名 | 作用 | 说明 |
|------|------|------|
| `{name}-enter-from` | 进入起始状态 | 元素插入前添加，插入后下一帧移除 |
| `{name}-enter-active` | 进入激活阶段 | 全程生效，定义过渡属性和时长 |
| `{name}-enter-to` | 进入结束状态 | 元素插入后下一帧添加 |
| `{name}-leave-from` | 离开起始状态 | 离开触发时添加 |
| `{name}-leave-active` | 离开激活阶段 | 全程生效，定义过渡属性和时长 |
| `{name}-leave-to` | 离开结束状态 | 离开触发后下一帧添加 |

### 2.2 默认类名（无 name）

当没有设置 `name` 属性时，默认前缀为 `v-`，即 `v-enter-from`、`v-leave-active` 等。

## 三、注意事项与常见陷阱

- `*-enter-from` 类只在元素插入前的**一帧**存在，时间极短
- `*-active` 类必须定义 `transition` 或 `animation` 属性才能看到效果
- 过渡结束由浏览器 `transitionend` / `animationend` 事件决定，不要手动移除类名
