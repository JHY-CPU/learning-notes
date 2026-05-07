# 封装 useFetch

## 一、概念说明

将请求逻辑封装为 Vue 组合式函数（Composable），实现请求状态（loading、data、error）的响应式管理，使组件代码更简洁。

```js
// composables/useFetch.js
import { ref, shallowRef } from 'vue'
import axios from 'axios'

export function useFetch(url, options = {}) {
  const data = shallowRef(null)
  const error = ref(null)
  const loading = ref(false)

  async function execute(params) {
    loading.value = true
    error.value = null
    try {
      const res = await axios({ url, ...options, params })
      data.value = res.data
      return res.data
    } catch (e) {
      error.value = e
      throw e
    } finally {
      loading.value = false
    }
  }

  // 自动请求（如果 immediate 不为 false）
  if (options.immediate !== false) {
    execute(options.params)
  }

  return { data, error, loading, execute, refresh: execute }
}
```

```vue
<script setup>
import { useFetch } from '@/composables/useFetch'

const { data: users, loading, error, refresh } = useFetch('/api/users')
</script>

<template>
  <button @click="refresh">刷新</button>
  <div v-if="loading">加载中...</div>
  <div v-else-if="error">错误: {{ error.message }}</div>
  <ul v-else>
    <li v-for="u in users" :key="u.id">{{ u.name }}</li>
  </ul>
</template>
```

## 二、具体用法

### 2.1 配置选项

```js
const { data, execute } = useFetch('/api/users', {
  method: 'GET',
  immediate: false,    // 不自动请求
  params: { page: 1 }
})
```

### 2.2 手动触发

```js
const { data, execute } = useFetch('/api/search', { immediate: false })

function search(keyword) {
  execute({ keyword })
}
```

### 2.3 shallowRef 优化

使用 `shallowRef` 而非 `ref` 存储响应数据，避免深度响应式转换大对象的性能开销。

## 三、注意事项与常见陷阱

- 使用 `shallowRef` 存储 API 返回的大对象，提升性能
- 组件卸载时如果请求未完成，需取消请求避免更新已销毁的 ref
- `immediate: false` 时必须手动调用 `execute`
