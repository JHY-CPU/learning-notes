# SSR性能优化

## 一、概念说明

SSR 性能优化关注两个核心指标：**TTFB**（首字节时间）和 **FCP**（首次内容绘制）。优化策略包括服务端缓存、CDN 分发、响应压缩、预加载关键资源等。Nuxt 3 的 Nitro 引擎内置了多层缓存机制。

## 二、具体用法

### 服务端缓存

```ts
// nuxt.config.ts
export default defineNuxtConfig({
  routeRules: {
    // 静态页面缓存1天
    '/about': { cache: { maxAge: 86400 } },

    // 产品页 SWR：缓存1小时，过期后异步更新
    '/products/**': { swr: 3600 },

    // API 响应缓存5分钟
    '/api/products': {
      cache: { maxAge: 300, staleMaxError: 1 }
    },

    // 首页预渲染
    '/': { prerender: true }
  }
})
```

### CDN 缓存头配置

```ts
// server/middleware/cache-headers.ts
export default defineEventHandler((event) => {
  const path = event.path

  if (path?.startsWith('/api/')) {
    // API 响应不缓存
    setResponseHeader(event, 'Cache-Control', 'no-store')
  } else if (path?.startsWith('/static/')) {
    // 静态资源长期缓存
    setResponseHeader(event, 'Cache-Control', 'public, max-age=31536000, immutable')
  } else {
    // HTML 页面短时间缓存
    setResponseHeader(event, 'Cache-Control', 'public, max-age=60, s-maxage=300')
  }
})
```

### 预加载关键资源

```vue
<!-- 使用 Nuxt 内置的预加载 -->
<script setup>
useHead({
  link: [
    // 预加载字体
    { rel: 'preload', href: '/fonts/main.woff2', as: 'font', crossorigin: '' },
    // 预连接 API 服务器
    { rel: 'preconnect', href: 'https://api.example.com' },
    // DNS 预解析
    { rel: 'dns-prefetch', href: 'https://cdn.example.com' }
  ]
})
</script>
```

### 响应压缩

```ts
// nuxt.config.ts
export default defineNuxtConfig({
  nitro: {
    compressPublicAssets: true,  // 压缩静态资源
    // 自动使用 gzip 或 brotli
  }
})
```

### 组件懒加载

```vue
<template>
  <div>
    <!-- 首屏组件正常加载 -->
    <HeroSection />
    <FeaturedProducts />

    <!-- 非首屏组件懒加载 -->
    <LazyUserReviews />
    <LazyRelatedProducts />
    <LazyNewsletter />
    <!-- 进入视口时才加载对应 JS -->
  </div>
</template>
```

### 性能监控

```vue
<script setup>
// 使用 useServerSeoMeta 避免不必要的 meta 注入
useServerSeoMeta({
  title: '我的页面',
  description: '页面描述'
})

// 追踪核心 Web 指标
onMounted(() => {
  new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) {
      console.log(`${entry.name}: ${entry.startTime}ms`)
      // LCP: 1200ms, FID: 50ms, CLS: 0.05
    }
  }).observe({ type: 'largest-contentful-paint', buffered: true })
})
</script>
```

## 三、注意事项与常见陷阱

1. **缓存策略要考虑数据新鲜度**：过度缓存导致用户看到过时数据
2. **gzip/brotli 压缩需要服务器支持**：某些 CDN 需要手动开启
3. **预加载不要过度**：预加载太多资源反而挤占带宽
4. **Lazy 组件只影响客户端加载**：SSR 时仍然会渲染所有组件
5. **监控生产环境性能**：开发模式下性能指标不准确
