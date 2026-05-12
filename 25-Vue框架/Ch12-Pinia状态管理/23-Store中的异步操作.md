# Store中的异步操作

## 一、概念说明

Pinia的actions支持异步操作，可直接使用`async/await`。管理API调用的loading和error状态是常见模式。

```js
export const useProductStore = defineStore('products', () => {
  const items = ref([])
  const loading = ref(false)
  const error = ref(null)
  const pagination = ref({ page: 1, total: 0 })

  const fetchProducts = async (page = 1) => {
    loading.value = true
    error.value = null
    try {
      const res = await api.getProducts({ page })
      items.value = res.data
      pagination.value = { page, total: res.total }
    } catch (e) {
      error.value = e.message
    } finally {
      loading.value = false
    }
  }

  return { items, loading, error, pagination, fetchProducts }
})
```

## 二、具体用法

### 组件中使用

```vue
<script setup>
import { useProductStore } from '@/stores/products'

const store = useProductStore()

onMounted(() => store.fetchProducts())

const loadMore = () => store.fetchProducts(store.pagination.page + 1)
</script>

<template>
  <p v-if="store.loading">加载中...</p>
  <p v-else-if="store.error">{{ store.error }}</p>
  <ul v-else>
    <li v-for="item in store.items" :key="item.id">{{ item.name }}</li>
  </ul>
  <button @click="loadMore" :disabled="store.loading">加载更多</button>
</template>
```

### 并行请求

```js
const initDashboard = async () => {
  const [users, orders, stats] = await Promise.all([
    api.getUsers(),
    api.getOrders(),
    api.getStats()
  ])
  state.users = users
  state.orders = orders
  state.stats = stats
}
```

## 四、竞态条件处理

```ts
export const useSearchStore = defineStore('search', () => {
  const results = ref([])
  const loading = ref(false)
  let requestId = 0  // 请求计数器

  const search = async (keyword: string) => {
    const currentId = ++requestId
    loading.value = true

    try {
      const data = await api.search(keyword)

      // 如果有更新的请求，忽略这个结果
      if (currentId !== requestId) return

      results.value = data
    } finally {
      if (currentId === requestId) {
        loading.value = false
      }
    }
  }

  // 或者使用 AbortController
  let controller: AbortController | null = null

  const searchWithAbort = async (keyword: string) => {
    controller?.abort()  // 取消上一次请求
    controller = new AbortController()

    loading.value = true
    try {
      const data = await api.search(keyword, { signal: controller.signal })
      results.value = data
    } catch (e) {
      if (!isAbortError(e)) throw e
    } finally {
      loading.value = false
    }
  }

  return { results, loading, search, searchWithAbort }
})
```

## 五、乐观更新

```ts
export const useTodoStore = defineStore('todos', () => {
  const items = ref([])

  const toggleTodo = async (id: number) => {
    // 乐观更新：先修改UI
    const todo = items.value.find(t => t.id === id)
    if (!todo) return
    const original = todo.done
    todo.done = !todo.done

    try {
      await api.updateTodo(id, { done: todo.done })
    } catch (e) {
      // 回滚
      todo.done = original
      throw e
    }
  }

  return { items, toggleTodo }
})
```

## 六、请求重试

```ts
async function fetchWithRetry(fn, retries = 3, delay = 1000) {
  for (let i = 0; i < retries; i++) {
    try {
      return await fn()
    } catch (e) {
      if (i === retries - 1) throw e
      await new Promise(r => setTimeout(r, delay * (i + 1)))
    }
  }
}

// Store 中使用
const fetchData = async () => {
  loading.value = true
  try {
    data.value = await fetchWithRetry(() => api.getData())
  } finally {
    loading.value = false
  }
}
```

## 三、注意事项与常见陷阱

1. 异步action返回Promise，可await
2. 注意竞态条件：快速连续调用可能数据错乱
3. 组件卸载时取消未完成的请求（AbortController）
4. loading状态避免闪烁：可加延时显示
5. 错误处理统一，避免静默失败
6. 乐观更新适合非关键操作，需要实现回滚
7. 请求重试适合网络不稳定场景，注意指数退避
