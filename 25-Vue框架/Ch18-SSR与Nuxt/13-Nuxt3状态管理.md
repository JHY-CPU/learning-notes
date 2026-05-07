# Nuxt3状态管理

## 一、概念说明

Nuxt 3 提供 `useState` 用于 SSR 友好的状态管理，数据在服务端渲染后序列化到 HTML，客户端 hydration 时恢复。对于复杂状态，推荐集成 Pinia。`useState` 适合简单的跨组件共享数据，Pinia 适合大型应用的状态管理。

## 二、具体用法

### useState 基本用法

```vue
<!-- composables/useCounter.ts -->
export function useCounter() {
  // useState 第一个参数是 key，用于 SSR 序列化
  return useState('counter', () => 0)
}
```

```vue
<!-- components/CounterDisplay.vue -->
<script setup>
const count = useCounter()
// 服务端和客户端共享同一个状态
// 服务端渲染时 count = 0
// 客户端 hydration 后 count 保持 0
</script>

<template>
  <div>
    <p>当前计数: {{ count }}</p>
    <button @click="count++">+1</button>
    <button @click="count--">-1</button>
  </div>
</template>
```

### 用户认证状态

```ts
// composables/useAuth.ts
export function useAuth() {
  const user = useState<{ name: string; avatar: string } | null>('auth-user', () => null)

  function login(userData: { name: string; avatar: string }) {
    user.value = userData
    // user.value = { name: '张三', avatar: '/avatars/zhang.png' }
  }

  function logout() {
    user.value = null
  }

  return { user, login, logout }
}
```

```vue
<script setup>
const { user, login, logout } = useAuth()
</script>

<template>
  <div v-if="user">
    <img :src="user.avatar" :alt="user.name" />
    <span>{{ user.name }}</span>
    <button @click="logout">退出登录</button>
  </div>
  <button v-else @click="login({ name: '张三', avatar: '/avatars/zhang.png' })">
    登录
  </button>
</template>
```

### 集成 Pinia

```ts
// stores/user.ts
export const useUserStore = defineStore('user', {
  state: () => ({
    name: '',
    email: '',
    isLoggedIn: false
  }),
  actions: {
    async fetchUser() {
      const { data } = await useFetch('/api/user')
      if (data.value) {
        this.name = data.value.name
        this.email = data.value.email
        this.isLoggedIn = true
      }
    }
  }
})
```

```vue
<script setup>
const userStore = useUserStore()

// Nuxt 中 Pinia 状态也会自动序列化/反序列化
onMounted(() => {
  if (!userStore.isLoggedIn) {
    userStore.fetchUser()
  }
})
</script>

<template>
  <div v-if="userStore.isLoggedIn">
    <h1>欢迎，{{ userStore.name }}</h1>
    <p>邮箱: {{ userStore.email }}</p>
  </div>
</template>
```

## 三、注意事项与常见陷阱

1. **useState 的 key 必须唯一**：不同 composable 使用相同 key 会导致状态覆盖
2. **状态仅在请求期间共享**：SSR 中不同请求的状态完全隔离
3. **不要存储不可序列化的值**：useState 的值会被 JSON.stringify，函数、DOM 节点等不可序列化
4. **Pinia 需要安装 @pinia/nuxt**：`npm install @pinia/nuxt` 并在 modules 中配置
5. **客户端专用状态用 useState 的 ref**：localStorage 等浏览器存储应在 onMounted 中读取
