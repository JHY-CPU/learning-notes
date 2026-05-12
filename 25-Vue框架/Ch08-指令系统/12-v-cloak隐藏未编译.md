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

## 五、现代构建工具中的 v-cloak

使用 Vite 或 Webpack 构建时，模板在构建阶段就已编译完成，HTML 中不会出现原始的 `{{ }}` 语法。因此：

```
构建工具（Vite/Webpack）→ 不需要 v-cloak
CDN 引入 Vue → 需要 v-cloak
```

### 5.1 CDN 引入时的完整用法
```html
<!DOCTYPE html>
<html>
<head>
  <style>
    [v-cloak] {
      display: none !important;
    }
    .loading-screen {
      position: fixed;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      background: white;
    }
  </style>
</head>
<body>
  <div id="app" v-cloak>
    <div class="loading-screen" v-show="!isReady">
      加载中...
    </div>
    <div v-show="isReady">
      {{ message }}
    </div>
  </div>
  <script src="https://unpkg.com/vue@3/dist/vue.global.js"></script>
  <script>
    const { createApp, ref, onMounted } = Vue
    createApp({
      setup() {
        const message = ref('Hello Vue!')
        const isReady = ref(false)
        onMounted(() => { isReady.value = true })
        return { message, isReady }
      }
    }).mount('#app')
  </script>
</body>
</html>
```

## 六、v-cloak 的替代方案

```vue
<!-- 方案一：使用 CSS 变量控制 -->
<style>
.app:not(.ready) { visibility: hidden; }
.app.ready { visibility: visible; transition: opacity 0.3s; }
</style>

<!-- 方案二：使用 Teleport 做全局加载 -->
<template>
  <Teleport to="body">
    <div v-if="!isReady" class="global-loading">加载中...</div>
  </Teleport>
</template>
```
