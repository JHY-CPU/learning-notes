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

## 四、路由名称类型安全

```ts
// router/names.ts
export const RouteNames = {
  HOME: 'Home',
  USER: 'User',
  USER_PROFILE: 'UserProfile'
} as const

export type RouteName = typeof RouteNames[keyof typeof RouteNames]

// 使用
import type { RouteName } from '@/router/names'

function navigateTo(name: RouteName) {
  router.push({ name })
}

navigateTo('Home')        // OK
navigateTo('Invalid')     // TypeScript 报错
```

## 五、useRoute 类型增强

```ts
// composables/useTypedRoute.ts
import { useRoute } from 'vue-router'
import { computed } from 'vue'

export function useTypedRoute<T extends Record<string, string>>() {
  const route = useRoute()

  const params = computed(() => route.params as T)
  const query = computed(() => route.query as Record<string, string | string[]>)

  return { route, params, query }
}

// 使用
const { params } = useTypedRoute<{ id: string }>()
console.log(params.value.id)  // 类型为 string
```

## 六、unplugin-vue-router 自动类型

```bash
npm install -D unplugin-vue-router
```

```ts
// vite.config.ts
import VueRouter from 'unplugin-vue-router/vite'

export default defineConfig({
  plugins: [
    VueRouter({
      routesFolder: 'src/views',
      dts: 'src/typed-router.d.ts'
    })
  ]
})
```

使用后，路由名称和参数会自动生成类型，无需手动声明：

```ts
router.push({ name: '/user/[id]', params: { id: '123' } })
// 自动推断 name 和 params 类型
```

## 三、注意事项与常见陷阱

1. `route.params`的值始终是`string`或`string[]`，需手动转换类型
2. 扩展`RouteMeta`接口需要创建`.d.ts`声明文件
3. 使用`definePage`宏可定义页面级元信息（需unplugin-vue-router）
4. 路由守卫的返回值类型：`boolean | string | RouteLocationRaw | void`
5. TypeScript严格模式下，守卫返回路径需符合`RouteLocationRaw`类型
6. `unplugin-vue-router`提供完整的类型推断，推荐在新项目中使用
7. 手动定义路由名称常量可以获得更精确的类型检查
