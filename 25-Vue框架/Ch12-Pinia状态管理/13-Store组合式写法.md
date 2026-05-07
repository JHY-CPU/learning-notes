# Store组合式写法

## 一、概念说明

Setup式Store使用setup函数（类似`<script setup>`）定义Store，推荐使用此方式。它更灵活，TypeScript支持更好。

```js
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useCounter = defineStore('counter', () => {
  // 定义state
  const count = ref(0)
  const name = ref('计数器')

  // 定义getter
  const doubled = computed(() => count.value * 2)
  const isPositive = computed(() => count.value > 0)

  // 定义action
  const increment = () => { count.value++ }
  const decrement = () => { count.value-- }
  const reset = () => { count.value = 0 }

  return {
    count, name,
    doubled, isPositive,
    increment, decrement, reset
  }
})
```

## 二、具体用法

### 完整示例

```js
export const useAuth = defineStore('auth', () => {
  const user = ref(null)
  const token = ref(localStorage.getItem('token'))
  const isLoggedIn = computed(() => !!token.value)
  const role = computed(() => user.value?.role || 'guest')

  const login = async (creds) => {
    const res = await api.login(creds)
    user.value = res.user
    token.value = res.token
    localStorage.setItem('token', res.token)
  }

  const logout = () => {
    user.value = null
    token.value = null
    localStorage.removeItem('token')
  }

  return { user, token, isLoggedIn, role, login, logout }
})
```

### 使用其他Store

```js
export const useOrder = defineStore('order', () => {
  const items = ref([])

  const checkout = async () => {
    const auth = useAuth()
    if (!auth.isLoggedIn) throw new Error('未登录')
    return api.createOrder(items.value, auth.token)
  }

  return { items, checkout }
})
```

## 三、注意事项与常见陷阱

1. Setup式没有`$reset()`，需自行实现
2. 返回值使用ref保持响应式，不要用reactive（解构会丢失）
3. 可以调用其他Store，Pinia处理好依赖
4. 组合式写法更接近`<script setup>`的风格
5. TypeScript推断更准确，IDE提示更好
