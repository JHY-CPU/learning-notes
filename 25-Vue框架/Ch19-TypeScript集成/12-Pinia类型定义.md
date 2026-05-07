# Pinia类型定义

## 一、概念说明

Pinia 天然支持 TypeScript，`defineStore` 的泛型语法和 Setup Store 模式都提供完整的类型推断。Store 的 state、getters、actions 自动获得类型。使用 `storeToRefs` 解构保持响应性和类型。

## 二、具体用法

### Setup Store（推荐）

```ts
// stores/counter.ts
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useCounterStore = defineStore('counter', () => {
  // 完整的类型推断
  const count = ref(0)      // Ref<number>
  const name = ref('计数器') // Ref<string>

  const doubled = computed(() => count.value * 2)
  // ComputedRef<number>

  function increment() {
    count.value++
  }

  function setCount(value: number) {
    count.value = value
  }

  return { count, name, doubled, increment, setCount }
})
// Store 类型自动从返回值推断
```

### Options Store

```ts
// stores/user.ts
import { defineStore } from 'pinia'

interface UserState {
  id: number | null
  name: string
  email: string
  role: 'admin' | 'editor' | 'viewer'
  isLoggedIn: boolean
}

export const useUserStore = defineStore('user', {
  state: (): UserState => ({
    id: null,
    name: '',
    email: '',
    role: 'viewer',
    isLoggedIn: false
  }),

  getters: {
    // 返回类型自动推断
    isAdmin: (state): boolean => state.role === 'admin',
    displayName: (state): string => state.name || '未登录'
  },

  actions: {
    async login(email: string, password: string) {
      // 模拟登录
      this.id = 1
      this.name = '张三'
      this.email = email
      this.role = 'admin'
      this.isLoggedIn = true
    },

    logout() {
      this.$reset()
    }
  }
})
```

### 在组件中使用

```vue
<script setup lang="ts">
import { storeToRefs } from 'pinia'
import { useCounterStore } from '@/stores/counter'
import { useUserStore } from '@/stores/user'

const counter = useCounterStore()
const user = useUserStore()

// storeToRefs 保持响应性和类型
const { count, doubled } = storeToRefs(counter)
// count 类型: Ref<number>
// doubled 类型: ComputedRef<number>

// actions 直接解构
const { increment, setCount } = counter
// increment 类型: () => void

async function handleLogin() {
  await user.login('zhang@example.com', '123456')
  console.log('登录成功:', user.displayName)
  // 输出：登录成功: 张三
}
</script>

<template>
  <div>
    <p>计数: {{ count }} (双倍: {{ doubled }})</p>
    <button @click="increment">+1</button>
    <button @click="setCount(100)">设为100</button>

    <div v-if="user.isLoggedIn">
      <p>欢迎, {{ user.displayName }} ({{ user.role }})</p>
      <button @click="user.logout()">退出</button>
    </div>
    <button v-else @click="handleLogin">登录</button>
  </div>
</template>
```

### Store 类型复用

```ts
// types/store.ts
import type { useUserStore } from '@/stores/user'

// 提取 Store 类型
export type UserStore = ReturnType<typeof useUserStore>

// 在需要 Store 类型的地方使用
function requireAuth(store: UserStore) {
  if (!store.isLoggedIn) {
    throw new Error('需要登录')
  }
}
```

## 三、注意事项与常见陷阱

1. **Setup Store 类型推断最好**：无需额外类型定义，推荐使用
2. **Options Store 需要 state 类型注释**：`state: (): UserState => ({...})`
3. **storeToRefs 只用于 state 和 getters**：actions 应直接从 store 解构
4. **$reset 在 Setup Store 中不可用**：需要手动实现重置逻辑
5. **跨 Store 引用保持类型**：`const user = useUserStore()` 在另一个 store 中也获得类型
