# 为什么需要SSR

## 一、概念说明

SSR 解决了 SPA（单页应用）的三大核心痛点：**SEO 不友好**、**首屏加载慢**、**社交分享预览缺失**。对于内容型网站、电商平台、博客等场景，SSR 几乎是刚需。

- **SEO（搜索引擎优化）**：搜索引擎爬虫更擅长解析完整的 HTML，而非执行 JavaScript
- **首屏性能**：用户无需等待 JS 下载和执行就能看到页面内容
- **社交分享**：Open Graph 和 Twitter Card 元标签需要在 HTML 中完整呈现

## 二、具体用法

### SEO 对比

```vue
<!-- CSR 问题：爬虫获取到的HTML几乎为空 -->
<!-- 爬虫看到的：<div id="app"></div> -->
<!-- 搜索引擎无法索引页面内容 -->

<!-- SSR 解决方案：服务器返回完整HTML -->
<script setup>
// 服务端设置 meta 标签
useHead({
  title: 'Vue.js 完整教程 - 从入门到精通',
  meta: [
    { name: 'description', content: '最全面的Vue.js中文教程' },
    { property: 'og:title', content: 'Vue.js 完整教程' },
    { property: 'og:description', content: '学习Vue 3组合式API' },
    { property: 'og:image', content: '/og-vue-tutorial.png' }
  ]
})
</script>

<template>
  <main>
    <h1>Vue.js 完整教程</h1>
    <section>
      <p>本文详细介绍 Vue 3 组合式 API 的使用方法...</p>
    </section>
  </main>
</template>
```

### 首屏性能对比

```js
// CSR 首屏加载流程（慢）
// 1. 浏览器请求 → 返回空HTML
// 2. 下载 JS bundle（通常 200KB+）
// 3. 执行 JS、初始化框架
// 4. 发起 API 请求获取数据
// 5. 渲染页面
// 总耗时：2-5秒

// SSR 首屏加载流程（快）
// 1. 浏览器请求 → 服务器渲染完整HTML
// 2. 浏览器立即显示内容
// 3. 后台下载 JS bundle
// 4. Hydration 激活交互
// 总耗时：0.5-1秒（首屏可见）
```

### 社交分享元标签

```vue
<script setup>
// Nuxt 3 中动态设置社交分享元信息
useServerSeoMeta({
  ogTitle: 'Vue SSR 实战指南',
  ogDescription: '深入了解 Vue 3 服务端渲染',
  ogImage: 'https://example.com/vue-ssr-cover.jpg',
  twitterCard: 'summary_large_image',
  twitterTitle: 'Vue SSR 实战指南'
})
</script>

<template>
  <article>
    <h1>Vue SSR 实战指南</h1>
    <p>服务端渲染让 Vue 应用更友好...</p>
  </article>
</template>
```

## 三、注意事项与常见陷阱

1. **并非所有网站都需要 SSR**：后台管理系统等内部工具用 CSR 即可
2. **SSR 增加服务器成本**：需要 Node.js 服务器，运维复杂度上升
3. **首屏快不等于交互快**：Hydration 阶段仍需 JS 执行，TTFB 可能变慢
4. **SEO 不是万能的**：Google 已能执行 JS，但百度等搜索引擎能力有限
5. **社交分享可用独立方案解决**：prerender-spa-plugin 可以只处理分享页面
