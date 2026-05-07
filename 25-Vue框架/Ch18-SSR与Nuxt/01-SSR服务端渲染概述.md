# SSR服务端渲染概述

## 一、概念说明

服务端渲染（Server-Side Rendering，SSR）是指在服务器端将 Vue 组件渲染为 HTML 字符串，然后将其发送到浏览器。与之相对的是客户端渲染（CSR），即浏览器下载空 HTML 后由 JavaScript 动态生成页面内容。

现代 Web 渲染策略主要有四种：

- **CSR（客户端渲染）**：浏览器获取空页面，JS 执行后渲染，SPA 默认模式
- **SSR（服务端渲染）**：服务器生成完整 HTML，客户端接管交互（hydration）
- **SSG（静态站点生成）**：构建时预渲染为静态 HTML，适合内容不变的页面
- **ISR（增量静态再生）**：SSG 的增强版，可定时重新生成静态页面

## 二、具体用法

### CSR vs SSR 对比

```vue
<!-- CSR: 页面首次加载时，用户看到的是空白或loading -->
<script setup>
import { ref, onMounted } from 'vue'

const posts = ref([])

// 数据在客户端获取，首次渲染无内容
onMounted(async () => {
  const res = await fetch('/api/posts')
  posts.value = await res.json()
})
</script>

<template>
  <div>
    <!-- 初始渲染时 posts 为空，页面空白 -->
    <article v-for="post in posts" :key="post.id">
      <h2>{{ post.title }}</h2>
      <p>{{ post.summary }}</p>
    </article>
  </div>
</template>
```

```vue
<!-- SSR: 服务器已渲染完整HTML，浏览器直接显示内容 -->
<script setup>
// 服务端执行：数据在渲染前就已准备好
const props = defineProps({
  posts: { type: Array, required: true }
})
</script>

<template>
  <div>
    <!-- 服务器返回的HTML已包含所有文章内容 -->
    <article v-for="post in posts" :key="post.id">
      <h2>{{ post.title }}</h2>
      <p>{{ post.summary }}</p>
    </article>
  </div>
</template>
```

### SSG 与 ISR 示例

```js
// Nuxt SSG: nuxi generate 预渲染所有页面
// nuxt.config.ts
export default defineNuxtConfig({
  nitro: {
    prerender: {
      routes: ['/about', '/contact', '/blog/1', '/blog/2']
    }
  }
})

// ISR: 每60秒重新验证页面
export default defineNuxtConfig({
  routeRules: {
    '/blog/**': { isr: 60 }    // 60秒后重新生成
  }
})
```

## 三、注意事项与常见陷阱

1. **CSR 首屏白屏问题**：JS 未加载完成前用户看到空白页面，影响用户体验
2. **SSR 需要 Node.js 运行环境**：不能像 CSR 一样部署到任意静态服务器
3. **SSG 只适合内容稳定的页面**：频繁变动的数据不适合纯静态生成
4. **ISR 需要平台支持**：Vercel、Netlify 原生支持，自建服务器需自行实现
5. **选择依据**：内容更新频率、SEO 需求、部署环境三者综合考虑
