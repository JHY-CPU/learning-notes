# SSR与Nuxt最佳实践

## 一、概念说明

选择正确的渲染策略、项目结构和开发规范是 Vue SSR 项目成功的关键。本文档总结了从技术选型到部署上线的最佳实践，涵盖常见问题的解决方案。

## 二、具体用法

### 渲染策略选择指南

```ts
// nuxt.config.ts - 推荐的混合渲染配置
export default defineNuxtConfig({
  routeRules: {
    // 营销页：SSG（最快加载）
    '/': { prerender: true },
    '/pricing': { prerender: true },

    // 博客：ISR（内容定期更新）
    '/blog/**': { isr: 3600 },

    // 产品：SSR（动态数据 + SEO）
    '/products/**': { swr: 600 },

    // 用户仪表盘：CSR（纯交互页面）
    '/dashboard/**': { ssr: false },

    // API 路由
    '/api/**': { cors: true }
  }
})

// 选择依据：
// 内容不变     → SSG / prerender
// 定期更新     → ISR / swr
// 实时数据     → SSR
// 纯交互应用  → CSR
```

### 项目结构规范

```
my-nuxt-app/
├── app.vue
├── pages/               # 文件路由
├── components/          # 按功能分组
│   ├── common/          # 通用组件
│   ├── product/         # 产品相关
│   └── layout/          # 布局组件
├── composables/         # 组合式函数，use 前缀
├── stores/              # Pinia stores
├── server/
│   ├── api/             # API 路由
│   └── middleware/      # 服务端中间件
├── middleware/          # 路由中间件
├── layouts/             # 布局模板
├── plugins/             # 插件
├── assets/              # 样式、字体
├── public/              # 静态文件
├── utils/               # 工具函数
└── nuxt.config.ts
```

### 常见问题解决

```vue
<!-- 问题1：Hydration 不匹配 -->
<script setup>
// 错误写法
const time = ref(Date.now())  // 服务端和客户端值不同

// 正确写法
const time = ref(0)
onMounted(() => {
  time.value = Date.now()  // hydration 后才赋值
})
</script>

<template>
  <div>
    <ClientOnly>
      <p>当前时间: {{ time }}</p>
    </ClientOnly>
  </div>
</template>
```

```ts
// 问题2：第三方库在 SSR 中报错
// plugins/leaflet.client.ts - 仅客户端加载地图库
export default defineNuxtPlugin(async () => {
  if (import.meta.client) {
    const L = await import('leaflet')
    // 安全使用浏览器专用库
    return { provide: { leaflet: L } }
  }
})
```

```vue
<!-- 问题3：闪烁问题 -->
<template>
  <!-- 使用 CSS 隐藏未 hydration 的内容 -->
  <div :class="{ 'hydrated': isHydrated }">
    <InteractiveWidget />
  </div>
</template>

<style>
.hydrated .interactive { visibility: visible; }
.interactive { visibility: hidden; }
</style>

<script setup>
const isHydrated = ref(false)
onMounted(() => { isHydrated.value = true })
</script>
```

### 开发检查清单

```text
1. [ ] 使用 useFetch/useAsyncData 获取数据（非 onMounted）
2. [ ] 浏览器 API 使用 ClientOnly 或 .client 后缀
3. [ ] 避免模板中的随机数和时间戳
4. [ ] 每个页面有唯一的 title 和 description
5. [ ] 图片使用 NuxtImg 并设置 width/height
6. [ ] 关键 CSS 内联，非关键 CSS 异步加载
7. [ ] 配置正确的 routeRules 渲染策略
8. [ ] 环境变量使用 NUXT_ 前缀
9. [ ] 部署前运行 nuxi build 测试
10. [ ] 使用 Lighthouse 检查性能和 SEO 分数
```

## 三、注意事项与常见陷阱

1. **不要在 SSR 项目中使用 localStorage**：用 useState 或 useCookie 替代
2. **onMounted 仅客户端执行**：不要在里面进行影响 SSR 输出的操作
3. **共享代码注意环境兼容**：utils/ 中的函数不能引用浏览器或 Node 独有 API
4. **Nuxt 3 的 auto-import 不适用于 node_modules**：第三方库仍需手动 import
5. **部署前必须生产构建**：`npm run dev` 的行为与生产环境不同
