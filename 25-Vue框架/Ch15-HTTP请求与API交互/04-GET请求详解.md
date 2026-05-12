# GET 请求详解

## 一、概念说明

GET 请求用于从服务器获取数据，参数通过 URL 查询字符串传递。在 Vue 应用中，GET 请求通常用于列表查询、详情获取等场景。

```vue
<script setup>
import { ref } from 'vue'
import axios from 'axios'

const users = ref([])
const query = ref({ page: 1, size: 10, keyword: '' })

async function fetchUsers() {
  const { data } = await axios.get('/api/users', { params: query.value })
  users.value = data.list
}

function nextPage() {
  query.value.page++
  fetchUsers()
}
</script>

<template>
  <input v-model="query.keyword" placeholder="搜索" @keyup.enter="fetchUsers" />
  <button @click="fetchUsers">查询</button>
  <button @click="nextPage">下一页</button>
  <ul><li v-for="u in users" :key="u.id">{{ u.name }}</li></ul>
</template>
```

## 二、具体用法

### 2.1 params 参数

```js
// GET /api/users?page=1&size=10&keyword=张三
axios.get('/api/users', {
  params: { page: 1, size: 10, keyword: '张三' }
})
```

### 2.2 paramsSerializer 自定义序列化

```js
import qs from 'qs'

axios.get('/api/users', {
  params: { tags: ['vue', 'react'] },
  paramsSerializer: params => qs.stringify(params, { arrayFormat: 'repeat' })
})
// 结果: /api/users?tags=vue&tags=react
```

### 2.3 带路径参数

```js
// GET /api/users/42
axios.get(`/api/users/${userId}`)
```

## 四、完整列表查询示例

```vue
<script setup>
import { ref, watch, onMounted } from 'vue'
import axios from 'axios'

const list = ref([])
const loading = ref(false)
const total = ref(0)
const query = ref({
  page: 1,
  pageSize: 20,
  keyword: '',
  status: '',
  sortBy: 'createdAt',
  sortOrder: 'desc'
})

async function fetchList() {
  loading.value = true
  try {
    const { data } = await axios.get('/api/articles', {
      params: {
        ...query.value,
        // 清除空值
        ...Object.fromEntries(
          Object.entries(query.value).filter(([_, v]) => v !== '' && v != null)
        )
      }
    })
    list.value = data.list
    total.value = data.total
  } finally {
    loading.value = false
  }
}

// 搜索时重置页码
watch(() => query.value.keyword, () => {
  query.value.page = 1
  fetchList()
})

onMounted(fetchList)
</script>

<template>
  <input v-model="query.keyword" placeholder="搜索..." />
  <select v-model="query.status">
    <option value="">全部状态</option>
    <option value="published">已发布</option>
    <option value="draft">草稿</option>
  </select>
  <button @click="fetchList" :disabled="loading">查询</button>

  <table>
    <tr v-for="item in list" :key="item.id">
      <td>{{ item.title }}</td>
      <td>{{ item.status }}</td>
    </tr>
  </table>

  <p>共 {{ total }} 条，第 {{ query.page }} 页</p>
  <button @click="query.page--; fetchList()" :disabled="query.page <= 1">上一页</button>
  <button @click="query.page++; fetchList()">下一页</button>
</template>
```

## 五、缓存 GET 请求

```js
const cache = new Map()

async function cachedGet(url, params) {
  const key = `${url}?${JSON.stringify(params)}`
  if (cache.has(key)) return cache[key]

  const { data } = await axios.get(url, { params })
  cache.set(key, data)

  // 5分钟后过期
  setTimeout(() => cache.delete(key), 5 * 60 * 1000)

  return data
}
```

## 三、注意事项与常见陷阱

- `params` 中值为 `null` 或 `undefined` 的参数会被自动忽略
- 数组参数的序列化方式需与后端约定
- GET 请求不支持 `body`，所有数据通过 URL 传递
- 搜索时应重置页码到第一页
- 缓存策略要考虑数据实时性要求
