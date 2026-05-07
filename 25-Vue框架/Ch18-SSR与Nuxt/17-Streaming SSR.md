# Streaming SSR

## 一、概念说明

流式 SSR（Streaming SSR）允许服务器将 HTML 分块发送给浏览器，而不是等待整个页面渲染完成。用户可以先看到页面框架和已就绪的内容，较慢的数据请求部分稍后通过流追加。结合 Vue 3 的 `<Suspense>` 组件，实现渐进式内容加载。

## 二、具体用法

### 基本流式渲染

```vue
<!-- pages/products.vue -->
<script setup>
// 慢速数据请求
const { data: products } = await useAsyncData('products', async () => {
  // 模拟耗时 2 秒的数据库查询
  await new Promise(r => setTimeout(r, 2000))
  return [
    { id: 1, name: 'Vue.js 实战', price: 59 },
    { id: 2, name: 'Nuxt 3 指南', price: 49 }
  ]
})
</script>

<template>
  <div>
    <h1>产品列表</h1>  <!-- 立即渲染 -->
    <Suspense>
      <template #default>
        <ProductList :products="products" />
      </template>
      <template #fallback>
        <div class="skeleton">加载产品中...</div>
        <!-- 数据未就绪时显示 -->
      </template>
    </Suspense>
  </div>
</template>
```

### Nuxt 3 流式 SSR 配置

```ts
// nuxt.config.ts
export default defineNuxtConfig({
  nitro: {
    routeRules: {
      '/': { swr: true }  // 结合流式和缓存
    }
  }
})
```

```vue
<!-- app.vue -->
<template>
  <div>
    <header>网站头部</header>  <!-- 第一块：立即发送 -->
    <NuxtLayout>
      <Suspense>
        <template #default>
          <NuxtPage />
        </template>
        <template #fallback>
          <LoadingSpinner />
        </template>
      </Suspense>
    </NuxtLayout>
    <footer>网站底部</footer>  <!-- 随后发送 -->
  </div>
</template>
```

### 嵌套 Suspense

```vue
<script setup>
// 多个独立数据请求，各自有独立加载状态
const { data: user } = await useFetch('/api/user')
const { data: orders } = await useLazyFetch('/api/orders')
const { data: recommendations } = await useLazyFetch('/api/recommendations')
</script>

<template>
  <div>
    <!-- 用户信息快速加载 -->
    <Suspense>
      <UserProfile :user="user" />
      <template #fallback>加载用户信息...</template>
    </Suspense>

    <!-- 订单数据较慢 -->
    <Suspense>
      <OrderList :orders="orders" />
      <template #fallback>加载订单中...</template>
    </Suspense>

    <!-- 推荐最慢 -->
    <Suspense>
      <Recommendations :items="recommendations" />
      <template #fallback>正在推荐...</template>
    </Suspense>
  </div>
</template>
```

## 三、注意事项与常见陷阱

1. **Suspense 仍是实验性特性**：Vue 3.4+ 的 Suspense 行为可能在后续版本调整
2. **流式 SSR 需要 HTTP/1.1 或 HTTP/2**：不支持分块传输的旧代理可能导致问题
3. **首字节时间（TTFB）变快**：但完全渲染时间不变，用户体验整体提升
4. **SEO 爬虫可能不等待流完成**：重要内容应放在首批发送的 HTML 中
5. **fallback 内容需有意义**：骨架屏优于空白，减少布局偏移（CLS）
