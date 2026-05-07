# Vue框架概述

## 一、概念说明

Vue.js 是一套用于构建用户界面的**渐进式 JavaScript 框架**。由尤雨溪（Evan You）于 2014 年发布，目前已发展到 Vue 3.x 版本。

**MVVM 模式**：Vue 采用了 Model-View-ViewModel 的架构模式。Model 是数据层，View 是视图层，ViewModel 是连接两者的桥梁。Vue 的响应式系统自动同步 Model 和 View，开发者只需关注数据逻辑。

**渐进式框架**意味着 Vue 的核心只关注视图层，可以逐步引入：从简单的 `<script>` 标签引入，到配合 Vue Router、Pinia 等官方工具构建大型单页应用（SPA）。

```vue
<script setup>
import { ref } from 'vue'

// 声明响应式数据
const message = ref('Hello Vue 3!')
const count = ref(0)
</script>

<template>
  <div>
    <h1>{{ message }}</h1>
    <p>计数: {{ count }}</p>
    <button @click="count++">+1</button>
  </div>
</template>

<style scoped>
h1 { color: #42b883; }
button { padding: 6px 16px; cursor: pointer; }
</style>
```

**Vue 2 vs Vue 3 主要区别**：Vue 3 使用 Proxy 替代 Object.defineProperty 实现响应式，性能更好；引入 Composition API 提升代码组织能力；支持 TypeScript 重写；Tree-shaking 更友好。

## 二、具体用法

### 2.1 渐进式采用

最简单的方式是通过 CDN 引入，适合小型项目或学习：

```html
<script src="https://unpkg.com/vue@3/dist/vue.global.js"></script>
<div id="app">{{ message }}</div>
<script>
  const { createApp, ref } = Vue
  createApp({
    setup() {
      return { message: ref('你好 Vue!') }
    }
  }).mount('#app')
</script>
```

### 2.2 Vue 核心特性

- **响应式数据绑定**：数据变化自动更新视图
- **组件化开发**：将 UI 拆分为独立可复用的组件
- **虚拟 DOM**：高效 diff 算法最小化真实 DOM 操作
- **指令系统**：v-if、v-for、v-model 等简化常见操作

## 三、注意事项与常见陷阱

- Vue 3 不再支持 IE11，需现代浏览器环境
- Vue 3 的响应式基于 ES6 Proxy，无法 polyfill
- 从 Vue 2 迁移需要关注破坏性变更（breaking changes）
- Vue 3 推荐使用 Composition API，但 Options API 仍然完全支持
- 学习 Vue 前建议掌握 HTML、CSS、JavaScript 基础
