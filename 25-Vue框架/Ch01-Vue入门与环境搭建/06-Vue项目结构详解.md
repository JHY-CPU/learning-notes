# Vue 项目结构详解

## 一、概念说明

Vue 3 + Vite 项目有标准化的目录结构。理解各目录的职责是高效开发的基础。`src/` 是源代码目录，`public/` 存放不需要构建的静态资源，配置文件位于项目根目录。

```vue
<script setup>
// main.js 入口文件示例
import { createApp } from 'vue'
import App from './App.vue'

const app = createApp(App)
// 注册全局组件、插件等
app.mount('#app')
</script>

<template>
  <!-- App.vue 根组件 -->
  <div id="app">
    <router-view />
  </div>
</template>
```

## 二、具体用法

### 2.1 核心目录结构

```
project/
├── src/
│   ├── assets/          # 静态资源（会被构建处理）
│   │   ├── images/      # 图片
│   │   └── styles/      # 全局样式
│   ├── components/      # 公共组件
│   │   └── Header.vue
│   ├── views/           # 页面级组件（路由页面）
│   │   └── Home.vue
│   ├── router/          # 路由配置
│   │   └── index.js
│   ├── stores/          # Pinia 状态管理
│   │   └── counter.js
│   ├── utils/           # 工具函数
│   ├── composables/     # 组合式函数
│   ├── App.vue          # 根组件
│   └── main.js          # 应用入口
├── public/              # 纯静态资源（直接复制）
│   └── favicon.ico
├── index.html           # HTML 入口
├── vite.config.js       # Vite 配置
└── package.json         # 项目信息和依赖
```

### 2.2 入口文件 main.js

```js
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import router from './router'
import App from './App.vue'
import './assets/styles/main.css'

const app = createApp(App)
app.use(createPinia())
app.use(router)
app.mount('#app')
```

### 2.3 组件目录规范

- `components/`：可复用的 UI 组件（Button、Modal、Card）
- `views/`：路由对应的页面组件（Home、About、UserDetail）
- `composables/`：可复用的组合式函数（useAuth、useFetch）

## 三、注意事项与常见陷阱

- `public/` 中的文件不会经过构建，适合放 favicon、robots.txt
- `assets/` 中的资源会被 Vite 处理（压缩、哈希），可通过 `import` 引用
- `@` 符号在 Vite 中默认指向 `src/` 目录
- 组件文件命名使用 PascalCase（如 `UserProfile.vue`）
- 避免在 `src/` 外创建 `.vue` 文件，Vite 不会处理它们
