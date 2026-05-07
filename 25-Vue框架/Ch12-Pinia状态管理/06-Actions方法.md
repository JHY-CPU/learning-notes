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
