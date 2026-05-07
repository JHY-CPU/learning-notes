# HTTP 请求概述

## 一、概念说明

现代 Web 应用普遍采用**前后端分离**架构。前端负责 UI 渲染和交互，后端提供数据接口（API）。Vue 应用通过 HTTP 请求与后端通信，获取和提交数据。

常见的 API 风格：
- **RESTful API**：基于 HTTP 方法（GET/POST/PUT/DELETE）操作资源
- **GraphQL**：客户端按需查询数据
- **WebSocket**：实时双向通信

```vue
<script setup>
import { ref, onMounted } from 'vue'

const users = ref([])
const loading = ref(false)
const error = ref(null)

onMounted(async () => {
  loading.value = true
  try {
    const res = await fetch('https://api.example.com/users')
    users.value = await res.json()
  } catch (e) {
    error.value = '请求失败'
  } finally {
    loading.value = false
  }
})
</script>

<template>
  <div v-if="loading">加载中...</div>
  <div v-else-if="error">{{ error }}</div>
  <ul v-else>
    <li v-for="user in users" :key="user.id">{{ user.name }}</li>
  </ul>
</template>
```

## 二、具体用法

### 2.1 RESTful API 四要素

| HTTP 方法 | 操作 | 示例 |
|-----------|------|------|
| GET | 查询 | `/api/users` |
| POST | 创建 | `/api/users` |
| PUT | 更新 | `/api/users/1` |
| DELETE | 删除 | `/api/users/1` |

### 2.2 请求流程

1. 浏览器发起请求 → 2. 服务器处理 → 3. 返回 JSON 响应 → 4. 前端解析渲染

## 三、注意事项与常见陷阱

- 跨域（CORS）问题需后端配置 `Access-Control-Allow-Origin`
- 敏感信息不要放在 URL 查询参数中
- 始终处理加载和错误状态，提升用户体验
