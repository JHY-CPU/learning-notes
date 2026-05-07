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

## 三、注意事项与常见陷阱

1. 异步action返回Promise，可await
2. 注意竞态条件：快速连续调用可能数据错乱
3. 组件卸载时取消未完成的请求（AbortController）
4. loading状态避免闪烁：可加延时显示
5. 错误处理统一，避免静默失败
