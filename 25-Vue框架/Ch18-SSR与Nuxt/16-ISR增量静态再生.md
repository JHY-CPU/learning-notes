# ISR增量静态再生

## 一、概念说明

ISR（Incremental Static Regeneration）是 SSG 的增强方案。页面在构建时或首次请求时生成静态 HTML，并设置一个过期时间。过期后，下一个请求会触发页面重新生成，期间仍提供旧页面。用户始终获得快速响应，数据又能保持相对新鲜。

## 二、具体用法

### 基本 ISR 配置

```ts
// nuxt.config.ts
export default defineNuxtConfig({
  routeRules: {
    '/products/**': {
      isr: 60  // 60秒后重新生成页面
    },
    '/blog/**': {
      isr: 3600  // 1小时后重新生成
    }
  }
})
```

### 按路由配置 ISR

```ts
// nuxt.config.ts
export default defineNuxtConfig({
  routeRules: {
    // 首页：每5分钟重新验证
    '/': { isr: 300 },

    // 产品详情页：每小时重新验证
    '/products/[id]': { isr: 3600 },

    // 用户页：纯 CSR，不使用 ISR
    '/user/profile': { isr: false },

    // API 路由：禁用缓存
    '/api/**': { isr: false }
  }
})
```

```vue
<!-- pages/products/[id].vue -->
<script setup>
const route = useRoute()
const { data: product } = await useFetch(`/api/products/${route.params.id})

// 产品数据每小时重新获取
// 用户看到的页面最多延迟1小时
// 浏览器响应头：X-Nextjs-Cache: STALE 或 HIT
</script>

<template>
  <div v-if="product">
    <h1>{{ product.name }}</h1>
    <p class="price">¥{{ product.price }}</p>
    <p>{{ product.description }}</p>
  </div>
</template>
```

### 手动重新验证

```ts
// server/api/revalidate.ts
export default defineEventHandler(async (event) => {
  // 接收 webhook 触发重新验证
  const { path, secret } = await readBody(event)

  if (secret !== process.env.REVALIDATE_SECRET) {
    throw createError({ statusCode: 401, statusMessage: '无效的 secret' })
  }

  // 使用 Nitro 的缓存重新验证
  // 实际使用取决于部署平台
  return { revalidated: true, path }
})
```

```bash
# 外部系统触发重新验证
curl -X POST https://your-site.com/api/revalidate \
  -H "Content-Type: application/json" \
  -d '{"path": "/products/123", "secret": "your-secret"}'
# 输出：{"revalidated":true,"path":"/products/123"}
```

### ISR 缓存行为示意

```text
时间线：0s -------- 60s -------- 120s
页面状态：FRESH     STALE       FRESH (重新生成)
用户请求：直接返回   返回旧页面    触发重新生成
          缓存HTML  后台更新      下次返回新页面
```

## 三、注意事项与常见陷阱

1. **ISR 需要平台支持**：Vercel 原生支持，自建 Node 服务器需要自行实现缓存逻辑
2. **过期不等于立即删除**：STALE 状态的页面仍然可用，后台异步更新
3. **首次访问可能较慢**：未预渲染的页面首次请求需要实时生成
4. **不适合实时数据**：股票价格、实时聊天等场景不适合 ISR
5. **调试时注意缓存**：开发模式下 ISR 不生效，需要生产构建测试
