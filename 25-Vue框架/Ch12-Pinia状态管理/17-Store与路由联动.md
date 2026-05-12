# Store与路由联动

## 一、概念说明

在路由守卫中使用Pinia store进行权限控制、数据预加载等。路由变化触发store更新，store状态影响路由行为。

```js
// router/guards.js
import { useAuthStore } from '@/stores/auth'

export function setupGuards(router) {
  router.beforeEach(async (to) => {
    const auth = useAuthStore()

    if (to.meta.requiresAuth && !auth.isLoggedIn) {
      return { path: '/login', query: { redirect: to.fullPath } }
    }
  })

  // 路由后置守卫：更新面包屑
  router.afterEach((to) => {
    const app = useAppStore()
    app.setBreadcrumb(to.matched.map(r => r.meta.title))
  })
}
```

## 二、具体用法

### 路由参数同步到Store

```vue
<script setup>
import { watch } from 'vue'
import { useRoute } from 'vue-router'
import { useProductStore } from '@/stores/products'

const route = useRoute()
const store = useProductStore()

watch(() => route.query.category, (category) => {
  store.setCategory(category)
}, { immediate: true })
</script>
```

### 导航守卫中获取数据

```js
router.beforeEach(async (to) => {
  const auth = useAuthStore()

  // 首次访问时获取用户信息
  if (auth.token && !auth.user) {
    await auth.fetchUser()
  }

  if (to.meta.requiresAuth && !auth.isLoggedIn) {
    return '/login'
  }
})
```

## 四、路由参数与Store同步

```js
// composables/useRouteStore.js
import { watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'

export function useRouteStore(store, fieldMap) {
  const route = useRoute()
  const router = useRouter()

  // 路由参数 → Store
  Object.entries(fieldMap).forEach(([queryKey, storeKey]) => {
    watch(
      () => route.query[queryKey],
      (val) => {
        if (val && store[storeKey] !== val) {
          store[storeKey] = val
        }
      },
      { immediate: true }
    )
  })

  // Store → 路由参数
  Object.entries(fieldMap).forEach(([queryKey, storeKey]) => {
    watch(
      () => store[storeKey],
      (val) => {
        if (val !== route.query[queryKey]) {
          router.replace({ query: { ...route.query, [queryKey]: val } })
        }
      }
    )
  })
}

// 使用
const store = useSearchStore()
useRouteStore(store, {
  q: 'keyword',
  page: 'page',
  sort: 'sortBy'
})
```

## 五、页面离开时保存Store状态

```js
router.beforeEach((to, from) => {
  // 离开搜索页时保存搜索状态
  if (from.path === '/search') {
    const search = useSearchStore()
    search.saveState()
  }

  // 进入搜索页时恢复状态
  if (to.path === '/search') {
    const search = useSearchStore()
    search.restoreState()
  }
})
```

## 三、注意事项与常见陷阱

1. 在路由守卫中调用store，确保Pinia已初始化
2. 不要在store中直接引用router（循环依赖）
3. 导航守卫中获取数据时注意异步和错误处理
4. 路由参数变化不触发组件重建时，需watch监听
5. Store中的数据可用于路由守卫的判断逻辑
6. 双向同步路由和Store时注意避免循环更新
7. 保存/恢复状态功能适合搜索、筛选等场景
