# Transition 组件基础

## 一、概念说明

`<Transition>` 是 Vue 内置的过渡包装组件，为**单个**元素或组件的进入/离开添加动画效果。它不会渲染为 DOM 元素，仅作为逻辑容器。

当被包裹的元素发生 `v-if` 切换或 `v-show` 切换时，Vue 会自动添加/移除对应的 CSS 类名，从而触发过渡动画。

```vue
<script setup>
import { ref } from 'vue'
const visible = ref(true)
</script>

<template>
  <button @click="visible = !visible">切换显示</button>
  <Transition>
    <div v-if="visible" class="box">内容区域</div>
  </Transition>
</template>

<style>
.v-enter-active { transition: all 0.3s ease; }
.v-leave-active { transition: all 0.3s ease; }
.v-enter-from { opacity: 0; transform: translateY(-20px); }
.v-leave-to { opacity: 0; transform: translateY(20px); }
</style>
```

## 二、具体用法

### 2.1 name 属性

设置 `name` 后，CSS 类名前缀会从 `v-` 变为 `name-`。例如 `name="fade"` 会生成 `fade-enter-active` 等类名。

```vue
<template>
  <Transition name="fade">
    <p v-if="show">淡入淡出效果</p>
  </Transition>
</template>

<style>
.fade-enter-active, .fade-leave-active {
  transition: opacity 0.5s;
}
.fade-enter-from, .fade-leave-to {
  opacity: 0;
}
</style>
```

### 2.2 mode 属性

- `mode="out-in"`：先执行离开动画，再执行进入动画
- `mode="in-out"`：先执行进入动画，再执行离开动画

```vue
<Transition name="fade" mode="out-in">
  <button v-if="isOn" @click="isOn = false">ON</button>
  <button v-else @click="isOn = true">OFF</button>
</Transition>
```

## 三、注意事项与常见陷阱

- `<Transition>` 内只能有**一个**直接子元素，多个元素会导致动画异常
- `v-show` 也能触发 `<Transition>`，但进入时不会触发初始渲染动画（需用 `appear`）
- 使用 `mode` 避免两个元素同时存在的闪烁问题
