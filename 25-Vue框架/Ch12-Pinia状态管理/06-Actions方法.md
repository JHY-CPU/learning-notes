# Actions方法

## 一、概念说明

Actions定义Store的方法，可包含同步和异步操作。Pinia没有mutations，所有状态修改都在actions中。

```js
export const useUserStore = defineStore('user', {
  state: () => ({
    user: null,
    loading: false,
    error: null
  }),
  actions: {
    // 同步action
    setUser(user) {
      this.user = user
    },
    logout() {
      this.user = null
      localStorage.removeItem('token')
    },

    // 异步action
    async login(credentials) {
      this.loading = true
      this.error = null
      try {
        const res = await api.login(credentials)
        this.user = res.data
        localStorage.setItem('token', res.token)
      } catch (e) {
        this.error = e.message
      } finally {
        this.loading = false
      }
    }
  }
})
```

## 二、具体用法

### 在组件中调用

```vue
<script setup>
import { useUserStore } from '@/stores/user'

const userStore = useUserStore()

const handleLogin = async () => {
  await userStore.login({ username: 'admin', password: '123456' })
  if (!userStore.error) {
    router.push('/dashboard')
  }
}
</script>
```

### action间互相调用

```js
actions: {
  async fetchData() {
    const configStore = useConfigStore()
    const url = configStore.apiUrl + '/data'
    this.data = await fetch(url).then(r => r.json())
  }
}
```

### Setup风格的action

```js
export const useUser = defineStore('user', () => {
  const user = ref(null)
  const login = async (creds) => {
    user.value = await api.login(creds)
  }
  return { user, login }
})
```

## 三、注意事项与常见陷阱

1. actions中可使用`this`访问整个store
2. 异步action返回Promise，可await
3. action中可调用其他store的action
4. 不要在组件中直接修改state，通过action修改（保持可追踪性）
5. action可以有返回值

## 四、Action 的高级用法

### 4.1 带错误处理的异步 Action
```js
export const useUserStore = defineStore('user', {
  state: () => ({
    user: null,
    loading: false,
    error: null
  }),
  actions: {
    async fetchUser(id) {
      this.loading = true
      this.error = null
      try {
        this.user = await api.getUser(id)
        return this.user
      } catch (e) {
        this.error = e.message
        throw e
      } finally {
        this.loading = false
      }
    }
  }
})
```

### 4.2 使用 $onAction 监听 Action 调用
```js
const store = useUserStore()

store.$onAction(({ name, args, after, onError }) => {
  const start = Date.now()

  after((result) => {
    console.log(`${name} 完成，耗时 ${Date.now() - start}ms`)
  })

  onError((error) => {
    console.error(`${name} 失败:`, error)
    // 上报错误
    reportError({ action: name, args, error })
  })
})
```

### 4.3 组合多个 Store 的 Action
```js
export const useOrderStore = defineStore('order', {
  actions: {
    async createOrder(items) {
      const cartStore = useCartStore()
      const userStore = useUserStore()

      if (!userStore.isLoggedIn) {
        throw new Error('请先登录')
      }

      const order = await api.createOrder({
        userId: userStore.user.id,
        items: cartStore.items
      })

      cartStore.clearCart()
      return order
    }
  }
})
```

## 五、Setup 风格的 Action

```js
export const useUser = defineStore('user', () => {
  const user = ref(null)
  const token = ref(localStorage.getItem('token'))

  async function login(credentials) {
    const result = await api.login(credentials)
    token.value = result.token
    user.value = result.user
    localStorage.setItem('token', result.token)
  }

  function logout() {
    token.value = null
    user.value = null
    localStorage.removeItem('token')
  }

  const isLoggedIn = computed(() => !!token.value)

  return { user, token, isLoggedIn, login, logout }
})
```
