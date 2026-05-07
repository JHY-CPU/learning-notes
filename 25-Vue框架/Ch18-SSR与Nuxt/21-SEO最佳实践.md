# SEO最佳实践

## 一、概念说明

SSR 为 SEO 提供了基础，但还需要正确的 meta 标签、结构化数据、语义化 HTML 和 sitemap 等配合。Nuxt 3 提供 `useHead`、`useSeoMeta`、`useServerSeoMeta` 等 API 来管理 SEO 相关配置。

## 二、具体用法

### useHead 设置 meta

```vue
<script setup>
useHead({
  title: 'Vue.js 入门教程 - 2024最新版',
  titleTemplate: '%s | 我的技术博客',
  // 最终标题：Vue.js 入门教程 - 2024最新版 | 我的技术博客

  meta: [
    { name: 'description', content: '最全面的 Vue 3 中文教程，覆盖组合式API、Pinia、Vue Router' },
    { name: 'keywords', content: 'Vue.js, Vue3, 前端教程, 组合式API' },
    { name: 'author', content: '张三' },
    { name: 'robots', content: 'index, follow' }
  ],

  htmlAttrs: { lang: 'zh-CN' },
  bodyAttrs: { class: 'dark-theme' }
})
</script>
```

### Open Graph 社交分享

```vue
<script setup>
useServerSeoMeta({
  // Open Graph 标签
  ogTitle: 'Vue.js 入门教程',
  ogDescription: '学习 Vue 3 组合式 API，从零开始构建现代 Web 应用',
  ogImage: 'https://example.com/images/vue-tutorial-cover.jpg',
  ogUrl: 'https://example.com/vue-tutorial',
  ogType: 'article',
  ogLocale: 'zh_CN',

  // Twitter Card
  twitterTitle: 'Vue.js 入门教程',
  twitterDescription: '最全面的 Vue 3 中文教程',
  twitterImage: 'https://example.com/images/vue-tutorial-cover.jpg',
  twitterCard: 'summary_large_image'
})
</script>
```

### 结构化数据（JSON-LD）

```vue
<script setup>
useHead({
  script: [
    {
      type: 'application/ld+json',
      children: JSON.stringify({
        '@context': 'https://schema.org',
        '@type': 'Article',
        headline: 'Vue.js 入门教程',
        author: { '@type': 'Person', name: '张三' },
        datePublished: '2024-01-15',
        image: 'https://example.com/images/vue-tutorial.jpg',
        description: '全面的 Vue 3 教程'
      })
    }
  ]
})
</script>
```

### Sitemap 生成

```ts
// nuxt.config.ts
export default defineNuxtConfig({
  modules: ['nuxt-simple-sitemap'],

  sitemap: {
    siteUrl: 'https://example.com',
    urls: [
      { loc: '/', changefreq: 'daily', priority: 1.0 },
      { loc: '/about', changefreq: 'monthly', priority: 0.8 },
      { loc: '/blog', changefreq: 'daily', priority: 0.9 }
    ]
  }
})
```

### robots.txt

```ts
// server/routes/robots.txt.ts
export default defineEventHandler(() => {
  return `User-agent: *
Allow: /
Disallow: /admin/
Disallow: /api/
Sitemap: https://example.com/sitemap.xml`
})
```

## 三、注意事项与常见陷阱

1. **useServerSeoMeta 只在 SSR 时执行**：CSR 页面需要用 useSeoMeta
2. **og:image 必须是完整 URL**：相对路径在社交平台上无法显示
3. **结构化数据需验证**：使用 Google Rich Results Test 检查格式
4. **不要过度堆砌 keywords**：搜索引擎已不依赖 keywords meta 排名
5. **每个页面应有唯一的 title 和 description**：重复内容会影响 SEO 评分
