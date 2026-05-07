# SSG静态站点生成

## 一、概念说明

SSG（Static Site Generation）在构建时将页面预渲染为静态 HTML 文件。与 SSR 不同，SSG 不需要运行时服务器，HTML 文件可直接部署到 CDN。适合内容更新频率低的网站：博客、文档站、产品介绍页。

## 二、具体用法

### 基本 SSG 配置

```ts
// nuxt.config.ts
export default defineNuxtConfig({
  ssr: true,
  nitro: {
    prerender: {
      routes: ['/', '/about', '/contact']
      // 构建时预渲染这些路由
    }
  }
})
```

```bash
# 执行静态生成
npx nuxi generate
# 构建过程：
# 1. 渲染每个路由为 HTML
# 2. 提取 CSS 和 JS
# 3. 生成 .output/public/ 目录
# 输出：Generated 3 pages in 2.5s
```

### 动态路由静态化

```ts
// nuxt.config.ts
export default defineNuxtConfig({
  nitro: {
    prerender: {
      crawlLinks: true,    // 自动发现链接并预渲染
      routes: [
        '/blog/my-first-post',
        '/blog/vue-guide',
        '/blog/nuxt-tutorial'
      ]
    }
  }
})
```

```ts
// server/api/posts.ts - 构建时提供数据
export default defineEventHandler(() => {
  return [
    { slug: 'my-first-post', title: '我的第一篇文章' },
    { slug: 'vue-guide', title: 'Vue 指南' }
  ]
})
```

### 使用 Nuxt Content 创建静态博客

```vue
<!-- pages/blog/[...slug].vue -->
<script setup>
// Nuxt Content 从 content/ 目录读取 Markdown 文件
const { data: post } = await useAsyncData('post', () =>
  queryContent(useRoute().path).findOne()
)
</script>

<template>
  <article>
    <h1>{{ post.title }}</h1>
    <ContentRenderer :value="post" />
  </article>
</template>
```

```bash
# content/blog/vue-guide.md 自动成为 /blog/vue-guide 页面
# 构建时所有 Markdown 文件都被预渲染为静态 HTML
```

### 部署到 GitHub Pages

```yaml
# .github/workflows/deploy.yml
name: Deploy SSG
on:
  push:
    branches: [main]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm install
      - run: npx nuxi generate
      - uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./.output/public
```

## 三、注意事项与常见陷阱

1. **SSG 不支持服务端动态内容**：每次用户请求拿到的是构建时的快照
2. **`crawlLinks: true` 可能导致构建时间过长**：大型站点建议手动指定路由列表
3. **API 路由在 SSG 中不可用**：静态文件无法执行服务端代码
4. **数据时效性问题**：构建后数据更新需要重新构建部署，可配合 ISR 解决
5. **404 页面需要特殊处理**：未预渲染的路由会返回 404，考虑使用 SPA 回退
