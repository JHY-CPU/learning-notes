# 路由类型vue-router

## 一、概念说明

vue-router 4 对 TypeScript 提供了全面支持，包括路由参数类型、路由元信息类型、导航守卫类型等。正确配置路由类型可以确保参数访问的类型安全。

## 二、具体用法

### 2.1 路由配置类型

```typescript
import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router';

// 路由配置的类型
const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'Home',
    component: () => import('../views/Home.vue'),
  },
  {
    path: '/users/:id',
    name: 'User',
    component: () => import('../views/User.vue'),
    props: true, // 将路由参数作为 props 传递
  },
  {
    path: '/dashboard',
    name: 'Dashboard',
    component: () => import('../views/Dashboard.vue'),
    meta: { requiresAuth: true, title: '仪表盘' },
  },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});
```

### 2.2 路由参数类型

```vue
<script setup lang="ts">
import { useRoute } from 'vue-router';

const route = useRoute();

// route.params 的类型是 Record<string, string | string[]>
// 需要手动断言或使用 props: true
const userId = route.params.id as string;

// 使用 props: true 更类型安全
const props = defineProps<{
  id: string;
}>();
</script>
```

### 2.3 路由元信息类型

```typescript
// 扩展路由元信息类型
declare module 'vue-router' {
  interface RouteMeta {
    requiresAuth?: boolean;
    title?: string;
    roles?: Array<'admin' | 'user'>;
    icon?: string;
  }
}

// 导航守卫中使用
router.beforeEach((to, from) => {
  // to.meta.requiresAuth — boolean | undefined
  // to.meta.title — string | undefined
  if (to.meta.requiresAuth && !isLoggedIn()) {
    return { name: 'Login', query: { redirect: to.fullPath } };
  }

  document.title = to.meta.title ?? '默认标题';
});
```

### 2.4 useRoute 与 useRouter

```vue
<script setup lang="ts">
import { useRoute, useRouter } from 'vue-router';

const route = useRoute();
const router = useRouter();

// 路由信息
console.log(route.path);      // string
console.log(route.params);    // Record<string, string | string[]>
console.log(route.query);     // Record<string, string | string[]>
console.log(route.meta);      // RouteMeta

// 导航方法
router.push('/home');
router.push({ name: 'User', params: { id: '123' } });
router.replace('/login');
router.go(-1);
router.back();
</script>
```

### 2.5 类型化路由参数

```typescript
// 方式一：定义路由参数类型
interface UserRouteParams {
  id: string;
}

// 方式二：使用 props: true + defineProps
// <script setup lang="ts">
// const props = defineProps<{ id: string }>();
// </script>

// 方式三：自定义 useRoute hook
function useTypedRoute<T extends Record<string, string>>() {
  const route = useRoute();
  return {
    ...route,
    params: route.params as T,
    query: route.query as T,
  };
}

// 使用
const route = useTypedRoute<{ id: string }>();
const id = route.params.id; // 类型安全，string
```

### 2.6 路由组件类型

```typescript
import type { RouteComponent } from 'vue-router';

// 组件懒加载的类型
const Home: RouteComponent = () => import('../views/Home.vue');

// 路由守卫类型
import type { NavigationGuardNext, RouteLocationNormalized } from 'vue-router';

const authGuard = (
  to: RouteLocationNormalized,
  from: RouteLocationNormalized,
  next: NavigationGuardNext
) => {
  if (to.meta.requiresAuth) {
    next('/login');
  } else {
    next();
  }
};
```

## 三、注意事项与常见陷阱

1. **`useRoute().params` 类型是 `string | string[]`**：需要断言或使用 props
2. **路由元信息需要 `declare module` 扩展**：否则 `meta` 是空对象
3. **`router.push` 的 `params` 要与 `name` 配合**：不能与 `path` 一起使用
4. **props: true 推荐使用**：比 `route.params` 类型更安全
5. **动态路由的类型**：参数始终是字符串，需要手动转换
