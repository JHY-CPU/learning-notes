# Router类型

## 一、概念说明

Vue Router 4 对 TypeScript 有完整支持。路由参数、路由元信息（meta）、导航守卫等都支持类型定义。通过扩展 `RouteRecordRaw` 和 `RouteMeta` 接口来获得类型安全的路由配置。

## 二、具体用法

### 路由配置类型

```ts
// router/index.ts
import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'home',
    component: () => import('@/views/HomeView.vue'),
    meta: { title: '首页', requiresAuth: false }
  },
  {
    path: '/user/:id',
    name: 'user',
    component: () => import('@/views/UserView.vue'),
    meta: { title: '用户详情', requiresAuth: true }
  },
  {
    path: '/login',
    name: 'login',
    component: () => import('@/views/LoginView.vue'),
    meta: { title: '登录', requiresAuth: false }
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
```

### 扩展路由 Meta 类型

```ts
// router/index.ts
declare module 'vue-router' {
  interface RouteMeta {
    title: string
    requiresAuth: boolean
    roles?: string[]
  }
}

// 之后访问 route.meta 自动有类型
// route.meta.title → string
// route.meta.requiresAuth → boolean
// route.meta.roles → string[] | undefined
```

### 组件中获取路由参数

```vue
<!-- views/UserView.vue -->
<script setup lang="ts">
import { useRoute } from 'vue-router'

const route = useRoute()

// 路由参数类型
const userId = route.params.id
// userId 类型: string | string[]

// 安全地获取单个参数
const id = Array.isArray(userId) ? userId[0] : userId
console.log('用户ID:', id)

// 路由元信息
const pageTitle = route.meta.title
// pageTitle 类型: string
const needsAuth = route.meta.requiresAuth
// needsAuth 类型: boolean
</script>

<template>
  <div>
    <h1>{{ route.meta.title }}</h1>
    <p>查看用户: {{ $route.params.id }}</p>
  </div>
</template>
```

### 编程式导航类型

```vue
<script setup lang="ts">
import { useRouter } from 'vue-router'

const router = useRouter()

function goToUser(id: number) {
  router.push({
    name: 'user',
    params: { id: id.toString() }
  })
  // 跳转到 /user/123
}

function goHome() {
  router.push({ name: 'home' })
}

// 带查询参数
function search(keyword: string) {
  router.push({
    path: '/search',
    query: { q: keyword, page: '1' }
  })
}
</script>
```

### 导航守卫类型

```ts
// router/guards.ts
import type { Router, RouteLocationNormalized } from 'vue-router'

export function setupGuards(router: Router) {
  router.beforeEach((to: RouteLocationNormalized, from) => {
    // to.meta 类型安全
    if (to.meta.requiresAuth) {
      const token = localStorage.getItem('token')
      if (!token) {
        return { name: 'login', query: { redirect: to.fullPath } }
      }
    }

    // 设置页面标题
    document.title = `${to.meta.title} - 我的应用`
  })
}
```

## 三、注意事项与常见陷阱

1. **route.params 总是 string**：数字参数需要手动转换
2. **路由名要保持一致**：push 中的 name 必须与路由配置中定义的完全匹配
3. **meta 类型需要 declare module 扩展**：不扩展则 meta 为 `Record<string, any>`
4. **useRoute 在 setup 中使用**：不能在普通函数中调用
5. **query 参数可能为 undefined**：使用时需要做空值处理
