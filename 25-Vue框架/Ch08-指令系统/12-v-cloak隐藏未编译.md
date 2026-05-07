# v-cloak 隐藏未编译

## 一、概念说明
`v-cloak` 用于在 Vue 编译完成前**隐藏未编译的 Mustache 标签**（如 `{{ message }}`）。它会在编译完成后自动移除，需配合 CSS 使用。

## 二、具体用法

### 2.1 基本用法
```html
<!-- index.html -->
<style>
  [v-cloak] {
    display: none;
  }
</style>

<div id="app" v-cloak>
  {{ message }}
  <!-- Vue 编译前显示空白，编译后显示数据 -->
</div>
```

### 2.2 配合加载状态
```html
<style>
  [v-cloak] {
    display: none;
  }
  .loading {
    /* 加载动画样式 */
  }
</style>

<div id="app" v-cloak>
  <div class="loading">加载中...</div>
  <!-- Vue 编译完成后显示实际内容 -->
  <p>{{ message }}</p>
</div>
```

### 2.3 在组件中使用（Vue 3 SFC）
```vue
<template>
  <div v-cloak>
    <p>{{ data }}</p>
  </div>
</template>

<style>
[v-cloak] {
  display: none;
}
</style>
```

### 2.4 在模板中使用
```vue
<template>
  <!-- 应用根元素 -->
  <div v-cloak :class="{ ready: isReady }">
    <p>{{ message }}</p>
  </div>
</template>
<script setup>
import { ref, onMounted } from 'vue'
const isReady = ref(false)
const message = ref('Hello')

onMounted(() => {
  isReady.value = true
})
</script>

<style>
[v-cloak] {
  display: none;
}
</style>
```

## 三、注意事项与常见陷阱
- 使用 v-cloak **必须配合 CSS** `[v-cloak] { display: none; }`
- Vue 3 中使用构建工具（Vite/Webpack）时，v-cloak 通常不需要
- 因为构建后的 HTML 已包含编译好的内容
- v-cloak 主要用于 CDN 引入 Vue 的场景
- 编译完成后 v-cloak attribute 会被自动移除
