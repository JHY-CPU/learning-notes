# 路由与Pinia配合

## 一、概念说明

在路由守卫中使用Pinia store进行状态管理，实现登录验证、权限控制等功能。需要在router配置之前初始化Pinia。

```js
// main.js
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import router from './router'

const app = createApp(App)
const pinia = createPinia()

app.use(pinia)  // 先初始化Pinia
app.use(router) // 再使用router
app.mount('#app')
```

```js
// router/index.js
import { useAuthStore } from '@/stores/auth'

router.beforeEach((to) => {
  const authStore = useAuthStore()

  if (to.meta.requiresAuth && !authStore.isLoggedIn) {
    return { path: '/login', query: { redirect: to.fullPath } }
  }
})
```

## 二、具体用法

### 完整权限示例

```js
// stores/auth.js
export const useAuthStore = defineStore('auth', () => {
  const user = ref(null)
  const token = ref(localStorage.getItem('token'))
  const isLoggedIn = computed(() => !!token.value)
  const role = computed(() => user.value?.role || 'guest')

  const login = async (creds) => { /* ... */ }
  const logout = () => { token.value = null; user.value = null }

  return { user, token, isLoggedIn, role, login, logout }
})
```

```js
// router/index.js
router.beforeEach(async (to) => {
  const auth = useAuthStore()

  if (to.meta.requiresAuth) {
    if (!auth.isLoggedIn) return '/login'
    if (to.meta.role && auth.role !== to.meta.role) return '/403'
  }
})
```

## 三、注意事项与常见陷阱

1. Pinia必须在router之前初始化，否则在守卫中无法使用store
2. 守卫中获取store是安全的，因为守卫在应用启动后才执行
3. 登录后使用`router.push(redirect)`跳转回原页面
4. store中不要直接引用router（循环依赖），在守卫中调用
5. 异步获取用户信息时，确保只请求一次（避免守卫重复触发）

## 四、store 中使用 router 的替代方案

```js
// ❌ 不推荐：store 中直接使用 router（循环依赖风险）
// stores/auth.js
import router from '@/router'  // 可能导致循环依赖

// ✅ 推荐：在组件/守卫中调用 router
// stores/auth.js
export const useAuthStore = defineStore('auth', () => {
  const token = ref(null)
  const login = async (creds) => {
    token.value = await api.login(creds)
    // 不在这里做路由跳转
  }
  const logout = () => { token.value = null }
  return { token, login, logout }
})

// 在组件中处理跳转
const router = useRouter()
const authStore = useAuthStore()
await authStore.login(credentials)
router.push(redirect)
```

## 五、store 与路由参数同步

```js
// stores/filters.js
export const useFilterStore = defineStore('filters', () => {
  const keyword = ref('')
  const category = ref('')
  const page = ref(1)

  // 从路由参数初始化
  function initFromRoute(route) {
    keyword.value = route.query.q || ''
    category.value = route.query.cat || ''
    page.value = Number(route.query.page) || 1
  }

  // 同步到路由
  function syncToRoute(router) {
    router.replace({
      query: {
        q: keyword.value || undefined,
        cat: category.value || undefined,
        page: page.value > 1 ? page.value : undefined
      }
    })
  }

  return { keyword, category, page, initFromRoute, syncToRoute }
})
```
