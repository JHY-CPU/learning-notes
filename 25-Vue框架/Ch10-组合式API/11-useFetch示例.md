# useFetch示例

## 一、概念说明

`useFetch`封装了数据请求逻辑，提供响应式的加载状态、错误处理和数据管理。它是最常用的组合式函数之一。

```vue
<template>
  <div>
    <p v-if="loading">加载中...</p>
    <p v-else-if="error" class="error">{{ error }}</p>
    <ul v-else-if="data">
      <li v-for="user in data" :key="user.id">{{ user.name }}</li>
    </ul>
    <button @click="refetch" :disabled="loading">重新加载</button>
  </div>
</template>

<script setup>
import { useFetch } from './composables/useFetch'

const { data, loading, error, refetch } = useFetch('https://api.example.com/users')
</script>
```

## 二、具体用法

### 基础版本实现

```js
// composables/useFetch.js
import { ref, isRef, watch } from 'vue'

export function useFetch(url) {
  const data = ref(null)
  const error = ref(null)
  const loading = ref(false)

  const fetchData = async () => {
    loading.value = true
    error.value = null
    try {
      const response = await fetch(isRef(url) ? url.value : url)
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      data.value = await response.json()
    } catch (e) {
      error.value = e.message
    } finally {
      loading.value = false
    }
  }

  // 支持响应式URL
  if (isRef(url)) {
    watch(url, fetchData)
  }

  fetchData()
  return { data, error, loading, refetch: fetchData }
}
```

### 进阶版本 - 支持选项

```js
export function useFetch(url, options = {}) {
  const { immediate = true, onSuccess, onError } = options
  const data = ref(null)
  const error = ref(null)
  const loading = ref(false)

  const execute = async () => {
    loading.value = true
    try {
      const res = await fetch(url)
      data.value = await res.json()
      onSuccess?.(data.value)
    } catch (e) {
      error.value = e.message
      onError?.(e)
    } finally {
      loading.value = false
    }
  }

  if (immediate) execute()
  return { data, error, loading, execute }
}
```

## 三、注意事项与常见陷阱

1. 请求URL可以是`ref`，URL变化时自动重新请求
2. 在组件卸载时应取消未完成的请求（使用AbortController）
3. 注意竞态条件：快速切换URL时，后发的请求可能先返回
4. 错误处理要区分网络错误和业务错误
5. 敏感数据的请求应通过后端代理，不要在前端暴露API Key
