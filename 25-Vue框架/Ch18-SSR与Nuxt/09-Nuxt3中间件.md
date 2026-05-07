# Nuxt3中间件

## 一、概念说明

Nuxt 3 中间件用于在路由导航前后执行逻辑，分为**路由中间件**和**命名中间件**。它们常用于鉴权检查、页面访问控制、数据预加载等场景。中间件在 `middleware/` 目录下定义，或在页面中通过 `definePageMeta` 指定。

## 二、具体用法

### 全局中间件

```ts
// middleware/auth.global.ts
// .global.ts 后缀表示全局中间件，所有路由都会执行
export default defineNuxtRouteMiddleware((to, from) => {
  const token = useCookie('token')

  // 检查是否需要登录
  if (to.path.startsWith('/dashboard') && !token.value) {
    console.log(`访问 ${to.path} 需要登录，当前无 token`)
    return navigateTo('/login')
    // 未登录时重定向到 /login
  }
})
```

### 命名中间件

```ts
// middleware/auth.ts
export default defineNuxtRouteMiddleware((to, from) => {
  const user = useUserStore()

  if (!user.isLoggedIn) {
    return navigateTo({
      path: '/login',
      query: { redirect: to.fullPath }
    })
    // 重定向到 /login?redirect=/dashboard
  }
})
```

```vue
<!-- pages/dashboard.vue -->
<script setup>
// 使用命名中间件
definePageMeta({
  middleware: 'auth'
})
</script>

<template>
  <div>仪表盘（仅登录用户可访问）</div>
</template>
```

### 多个中间件

```vue
<!-- pages/admin/users.vue -->
<script setup>
// 按顺序执行多个中间件
definePageMeta({
  middleware: ['auth', 'admin']
})
// 执行顺序：auth → admin → 页面渲染
</script>

<template>
  <div>用户管理页面（需管理员权限）</div>
</template>
```

```ts
// middleware/admin.ts
export default defineNuxtRouteMiddleware(() => {
  const user = useUserStore()

  if (user.role !== 'admin') {
    throw createError({
      statusCode: 403,
      statusMessage: '无管理员权限'
    })
  }
})
```

### 匿名中间件（页面内）

```vue
<!-- 直接在页面中定义中间件 -->
<script setup>
definePageMeta({
  middleware: (to, from) => {
    // 页面级匿名中间件
    console.log(`从 ${from.path} 导航到 ${to.path}`)
    // 控制台输出：从 / 导航到 /settings
  }
})
</script>

<template>
  <div>设置页面</div>
</template>
```

## 三、注意事项与常见陷阱

1. **全局中间件用 `.global.ts` 后缀**：普通文件需手动在 `definePageMeta` 中指定才会执行
2. **中间件必须返回 undefined 或导航结果**：返回 false 不会阻止导航，需用 `navigateTo()`
3. **中间件在 SSR 时也会执行**：服务端无法使用 `localStorage` 等浏览器 API
4. **命名中间件只在 pages 中生效**：layouts 和 components 中设置无效
5. **中间件中不要进行重副作用**：中间件应轻量，耗时操作放在 `useAsyncData` 中
