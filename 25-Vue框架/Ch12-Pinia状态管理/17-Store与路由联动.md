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

## 三、注意事项与常见陷阱

1. 在路由守卫中调用store，确保Pinia已初始化
2. 不要在store中直接引用router（循环依赖）
3. 导航守卫中获取数据时注意异步和错误处理
4. 路由参数变化不触发组件重建时，需watch监听
5. Store中的数据可用于路由守卫的判断逻辑
