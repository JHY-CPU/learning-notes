# 手动搭建Vue SSR

## 一、概念说明

不依赖 Nuxt，手动搭建 Vue SSR 需要配置服务端入口、客户端入口和构建脚本。使用 Express 作为 HTTP 服务器，`vue-server-renderer` 在服务端将组件渲染为 HTML。这种方式帮助理解 SSR 底层原理，但生产环境推荐使用 Nuxt。

## 二、具体用法

### 项目结构

```
vue-ssr-demo/
├── src/
│   ├── App.vue          # 根组件
│   ├── entry-client.js  # 客户端入口
│   ├── entry-server.js  # 服务端入口
│   └── components/
├── server.js            # Express 服务器
├── index.html           # HTML 模板
├── vite.config.js       # Vite 构建配置
└── package.json
```

### 核心组件

```vue
<!-- src/App.vue -->
<script setup>
import { ref } from 'vue'

const message = ref('Vue SSR 手动搭建')
const count = ref(0)
</script>

<template>
  <div id="app">
    <h1>{{ message }}</h1>
    <p>计数器: {{ count }}</p>
    <button @click="count++">增加</button>
  </div>
</template>

<style scoped>
h1 { color: #42b883; }
button { padding: 8px 16px; cursor: pointer; }
</style>
```

### 服务端入口

```js
// src/entry-server.js
import { createSSRApp } from 'vue'
import { renderToString } from 'vue/server-renderer'
import App from './App.vue'

export function render() {
  const app = createSSRApp(App)
  // 返回渲染的 HTML 字符串
  return renderToString(app)
}
```

### 客户端入口

```js
// src/entry-client.js
import { createSSRApp } from 'vue'
import App from './App.vue'

// 客户端激活：复用服务端渲染的DOM
const app = createSSRApp(App)
app.mount('#app')
// 页面控制台输出：hydration 完成，按钮可点击
```

### Express 服务器

```js
// server.js
import express from 'express'
import { readFileSync } from 'fs'
import { render } from './src/entry-server.js'

const app = express()
const template = readFileSync('./index.html', 'utf-8')

// 静态资源
app.use(express.static('dist/client'))

app.get('*', async (req, res) => {
  try {
    const html = await render()
    // 将渲染结果注入模板
    const output = template.replace('<div id="app"></div>',
      `<div id="app">${html}</div>`)
    res.setHeader('Content-Type', 'text/html')
    res.end(output)
  } catch (e) {
    console.error(e)
    res.status(500).end('Internal Server Error')
  }
})

app.listen(3000, () => {
  console.log('SSR 服务运行在 http://localhost:3000')
})
// 终端输出：SSR 服务运行在 http://localhost:3000
```

### Vite SSR 构建配置

```js
// vite.config.js
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  build: {
    rollupOptions: {
      // 分别构建客户端和服务端
      input: {
        client: './src/entry-client.js',
        server: './src/entry-server.js'
      }
    }
  }
})
```

## 三、注意事项与常见陷阱

1. **每次请求创建新 App 实例**：避免在 Express 中复用 Vue 应用实例，否则用户间会共享状态
2. **构建产物分离**：客户端和服务端代码需要分别打包，服务端不能包含浏览器 API
3. **数据预取需要额外处理**：手动搭建需要自行实现 `onServerPrefetch` 的数据传递机制
4. **CSS 处理**：服务端渲染时无 DOM，需要提取 CSS 并注入 HTML
5. **生产环境推荐 Nuxt**：手动方案适合学习原理，生产项目直接用 Nuxt 省去大量配置
