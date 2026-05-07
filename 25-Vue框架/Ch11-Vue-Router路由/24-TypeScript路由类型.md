# TypeScript路由类型

## 一、概念说明

Vue Router 4提供完整的TypeScript支持，路由配置、守卫参数、组件中的`useRoute`/`useRouter`都有类型推断。

```ts
import type { RouteRecordRaw, RouteLocationRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'Home',
    component: Home
  },
  {
    path: '/user/:id',
    name: 'User',
    component: User,
    props: true
  }
]
```

## 二、具体用法

### 路由配置类型

```ts
import type { RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  {
    path: '/dashboard',
    name: 'Dashboard',
    component: () => import('@/views/Dashboard.vue'),
    meta: { requiresAuth: true }
  }
]
```

### 扩展meta类型

```ts
// router.d.ts 或 shims-vue-router.d.ts
import 'vue-router'

declare module 'vue-router' {
  interface RouteMeta {
    requiresAuth?: boolean
    title?: string
    roles?: string[]
    layout?: 'default' | 'blank'
  }
}
```

### 组件中使用类型

```vue
<script setup lang="ts">
import { useRoute, useRouter } from 'vue-router'

const route = useRoute()
const router = useRouter()

// route.params.id 类型为 string | string[]
const userId = route.params.id

// router.push 接受 RouteLocationRaw
router.push({ name: 'User', params: { id: '123' } })
</script>
```

### props类型

```vue
<script setup lang="ts">
interface Props {
  id: string
}

const props = defineProps<Props>()
console.log(props.id)  // string类型
</script>
```

## 三、注意事项与常见陷阱

1. `route.params`的值始终是`string`或`string[]`，需手动转换类型
2. 扩展`RouteMeta`接口需要创建`.d.ts`声明文件
3. 使用`definePage`宏可定义页面级元信息（需unplugin-vue-router）
4. 路由守卫的返回值类型：`boolean | string | RouteLocationRaw | void`
5. TypeScript严格模式下，守卫返回路径需符合`RouteLocationRaw`类型
