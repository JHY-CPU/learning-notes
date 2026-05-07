# Vue SSR原理

## 一、概念说明

Vue SSR 的核心流程分为两步：**服务端渲染**（renderToString）和 **客户端激活**（hydration）。服务端将 Vue 组件树渲染为 HTML 字符串发送给浏览器，浏览器展示 HTML 后，客户端 Vue 接管页面，将静态 HTML "激活"为可交互的动态应用。

- **renderToString**：在 Node.js 中将 Vue 组件实例化并序列化为 HTML 字符串
- **Hydration**：客户端 Vue 在已有的 DOM 上"附着"事件和响应式数据，而非重新创建 DOM

## 二、具体用法

### renderToString 基本原理

```js
// 服务端核心：renderToString
import { createSSRApp } from 'vue'
import { renderToString } from 'vue/server-renderer'
import App from './App.vue'

// 为每个请求创建新的应用实例（避免状态污染）
async function render(url) {
  const app = createSSRApp(App)

  // 将 Vue 应用渲染为 HTML 字符串
  const html = await renderToString(app)
  // html 输出示例：
  // '<div data-v-app=""><h1>你好世界</h1><p>计数: 0</p></div>'

  return `
    <!DOCTYPE html>
    <html>
    <head><title>Vue SSR</title></head>
    <body>
      <div id="app">${html}</div>
      <script src="/client.js"></script>
    </body>
    </html>
  `
}
```

### 客户端 Hydration

```js
// 客户端入口：client.js
import { createSSRApp } from 'vue'
import App from './App.vue'

// createSSRApp 自动启用 hydration 模式
// Vue 会复用已有的 DOM，只附加事件监听和响应式数据
const app = createSSRApp(App)
app.mount('#app')

// hydration 过程：
// 1. Vue 遍历现有 DOM 树
// 2. 对比虚拟 DOM 与实际 DOM
// 3. 绑定事件处理器
// 4. 建立响应式连接
// 5. DOM 完全复用，不重新创建
```

### 服务端与客户端渲染对比

```vue
<script setup>
import { ref } from 'vue'

const count = ref(0)

// 注意：onMounted 在服务端不会执行
import { onMounted } from 'vue'
onMounted(() => {
  console.log('仅在客户端执行')
  // 浏览器控制台输出：仅在客户端执行
})
</script>

<template>
  <div>
    <p>计数: {{ count }}</p>
    <button @click="count++">+1</button>
  </div>
</template>

<!-- 服务端输出HTML: <div><p>计数: 0</p><button>+1</button></div> -->
<!-- 客户端hydration后：按钮点击事件激活，count变为响应式 -->
```

## 三、注意事项与常见陷阱

1. **每个请求必须创建新实例**：共享应用实例会导致状态污染，所有用户看到相同数据
2. **服务端无 DOM API**：不能使用 `window`、`document`、`localStorage` 等浏览器 API
3. **onMounted 不会在服务端执行**：SSR 中的副作用应放在 `onServerPrefetch` 或条件判断中
4. **Hydration 不匹配会导致警告**：服务端和客户端渲染结果必须一致
5. **避免随机数和时间戳**：`Math.random()`、`Date.now()` 在服务端和客户端结果不同会导致不匹配
