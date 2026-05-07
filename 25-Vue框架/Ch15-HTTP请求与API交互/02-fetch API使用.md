# fetch API 使用

## 一、概念说明

`fetch` 是浏览器原生提供的 HTTP 请求 API，返回 Promise。无需安装第三方库，适合简单场景。

```vue
<script setup>
import { ref } from 'vue'

const data = ref(null)

// GET 请求
async function fetchData() {
  const res = await fetch('https://api.example.com/posts')
  data.value = await res.json()
}

// POST 请求
async function createPost() {
  const res = await fetch('https://api.example.com/posts', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title: '新文章', body: '内容' })
  })
  const result = await res.json()
  console.log('创建成功:', result)
}
</script>

<template>
  <button @click="fetchData">获取数据</button>
  <button @click="createPost">创建文章</button>
  <pre v-if="data">{{ JSON.stringify(data, null, 2) }}</pre>
</template>
```

## 二、具体用法

### 2.1 GET 请求

```js
const res = await fetch('/api/users')
if (!res.ok) throw new Error(`HTTP ${res.status}`)
const users = await res.json()
```

### 2.2 POST 请求

```js
const res = await fetch('/api/users', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ name: '张三', age: 25 })
})
```

### 2.3 错误处理

```js
try {
  const res = await fetch('/api/data')
  if (!res.ok) {
    // fetch 不会对 4xx/5xx 抛出错误，需手动检查
    throw new Error(`请求失败: ${res.status}`)
  }
  const data = await res.json()
} catch (e) {
  console.error('网络错误或解析失败:', e)
}
```

### 2.4 取消请求

```js
const controller = new AbortController()
fetch('/api/data', { signal: controller.signal })
// 取消
controller.abort()
```

## 三、注意事项与常见陷阱

- `fetch` **不会**对 HTTP 4xx/5xx 错误抛出异常，必须检查 `res.ok`
- `fetch` 在网络错误（如断网）时才会 reject
- 默认不携带 cookie，需要设置 `credentials: 'include'`
