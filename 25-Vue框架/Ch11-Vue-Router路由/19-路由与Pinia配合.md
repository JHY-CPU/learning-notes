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
