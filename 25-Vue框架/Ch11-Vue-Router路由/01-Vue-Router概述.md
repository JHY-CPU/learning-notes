# Vue-Router概述

## 一、概念说明

Vue Router是Vue.js的官方路由管理器，用于构建**单页面应用 (SPA)**。它将URL路径映射到组件，实现页面切换不刷新的效果。

```js
// main.js
import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import App from './App.vue'
import Home from './views/Home.vue'
import About from './views/About.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', component: Home },
    { path: '/about', component: About }
  ]
})

createApp(App).use(router).mount('#app')
```

## 二、具体用法

### 安装

```bash
npm install vue-router@4
```

### 基础结构

```
src/
  views/          # 路由页面组件
    Home.vue
    About.vue
  router/         # 路由配置
    index.js
  App.vue         # <RouterView />
```

```vue
<!-- App.vue -->
<template>
  <nav>
    <RouterLink to="/">首页</RouterLink>
    <RouterLink to="/about">关于</RouterLink>
  </nav>
  <RouterView />  <!-- 路由组件渲染位置 -->
</template>
```

### 路由模式

| 模式 | 说明 | URL示例 |
|------|------|---------|
| `createWebHistory` | HTML5 History | `/about` |
| `createWebHashHistory` | Hash模式 | `/#/about` |
| `createMemoryHistory` | SSR/测试用 | - |

## 三、注意事项与常见陷阱

1. Vue Router 4专用于Vue 3，Vue 2使用Vue Router 3
2. History模式需要服务器配置，刷新会404（需配置fallback）
3. Hash模式无需服务器配置，但URL带有`#`
4. 路由组件应放在`views/`目录，普通组件放在`components/`
5. 每个路由映射一个组件，组件在路由切换时被销毁/重建
