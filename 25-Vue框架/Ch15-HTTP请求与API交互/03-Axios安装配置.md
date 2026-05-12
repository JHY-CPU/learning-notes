# Axios 安装配置

## 一、概念说明

Axios 是最流行的 HTTP 请求库，相比原生 `fetch`，它提供了请求/响应拦截器、自动 JSON 转换、请求取消、超时控制等增强功能。

```vue
<script setup>
import axios from 'axios'
import { ref } from 'vue'

const users = ref([])

async function fetchUsers() {
  const { data } = await axios.get('https://api.example.com/users')
  users.value = data
}
</script>

<template>
  <button @click="fetchUsers">获取用户</button>
  <ul><li v-for="u in users" :key="u.id">{{ u.name }}</li></ul>
</template>
```

## 二、具体用法

### 2.1 安装

```bash
npm install axios
```

### 2.2 创建实例

```js
// utils/request.js
import axios from 'axios'

const request = axios.create({
  baseURL: 'https://api.example.com',
  timeout: 10000,
  headers: { 'Content-Type': 'application/json' }
})

export default request
```

### 2.3 基本配置项

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `baseURL` | 请求基础路径 | 无 |
| `timeout` | 超时时间（ms） | 0（无限） |
| `headers` | 请求头 | `{}` |
| `params` | URL 查询参数 | `{}` |
| `withCredentials` | 携带 cookie | `false` |

### 2.4 常用方法

```js
axios.get('/users', { params: { page: 1 } })
axios.post('/users', { name: '张三' })
axios.put('/users/1', { name: '李四' })
axios.delete('/users/1')
axios.patch('/users/1', { age: 26 })
```

## 四、环境变量配置

```js
// .env.development
VITE_API_BASE=http://localhost:3000/api

// .env.production
VITE_API_BASE=https://api.example.com

// utils/request.js
const request = axios.create({
  baseURL: import.meta.env.VITE_API_BASE,
  timeout: 10000
})
```

## 五、完整的请求实例配置

```js
// utils/request.js
import axios from 'axios'
import { useUserStore } from '@/stores/user'

const request = axios.create({
  baseURL: import.meta.env.VITE_API_BASE,
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截
request.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => Promise.reject(error)
)

// 响应拦截
request.interceptors.response.use(
  (response) => response.data,
  (error) => {
    if (error.response?.status === 401) {
      // Token 过期，跳转登录
      const userStore = useUserStore()
      userStore.logout()
    }
    return Promise.reject(error)
  }
)

export default request
```

## 三、注意事项与常见陷阱

- 建议始终创建 `axios` 实例而非直接使用全局 `axios`
- `baseURL` 末尾不要加 `/`，避免路径拼接问题
- 生产环境通过环境变量配置 `baseURL`：`import.meta.env.VITE_API_BASE`
- `timeout` 设置过短可能导致正常请求被截断
- `withCredentials: true` 用于跨域携带 cookie，需要后端配合 CORS 配置
