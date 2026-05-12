# SSE 服务器推送

## 一、概念说明

SSE（Server-Sent Events）允许服务器向浏览器**单向推送**实时数据。相比 WebSocket，SSE 更简单，基于 HTTP 协议，自动重连，适合通知、实时数据更新等场景。

```vue
<script setup>
import { ref, onUnmounted } from 'vue'

const messages = ref([])
let eventSource = null

function connect() {
  eventSource = new EventSource('/api/events')

  eventSource.onmessage = (event) => {
    messages.value.push(JSON.parse(event.data))
  }

  eventSource.onerror = () => {
    console.error('SSE 连接错误')
  }
}

function disconnect() {
  eventSource?.close()
}

onUnmounted(disconnect)
</script>

<template>
  <button @click="connect">开始监听</button>
  <button @click="disconnect">断开连接</button>
  <ul>
    <li v-for="(msg, i) in messages" :key="i">{{ msg.text }}</li>
  </ul>
</template>
```

## 二、具体用法

### 2.1 EventSource 基本用法

```js
const es = new EventSource('/api/events')

// 监听默认消息
es.onmessage = (e) => console.log(e.data)

// 监听自定义事件
es.addEventListener('notification', (e) => {
  console.log(JSON.parse(e.data))
})
```

### 2.2 带认证的 SSE

```js
// EventSource 不支持自定义 header，使用 fetch 替代
const res = await fetch('/api/events', {
  headers: { Authorization: `Bearer ${token}` }
})
const reader = res.body.getReader()
// 手动解析 SSE 数据流...
```

### 2.3 事件类型

```js
es.addEventListener('alert', (e) => { /* 处理 alert 事件 */ })
es.addEventListener('update', (e) => { /* 处理 update 事件 */ })
es.addEventListener('error', (e) => { /* 处理自定义 error 事件 */ })
```

## 四、完整通知组件

```vue
<script setup>
import { ref, onUnmounted, watch } from 'vue'

const notifications = ref([])
const connected = ref(false)
let eventSource = null

function connect(token) {
  eventSource = new EventSource(`/api/events?token=${token}`)

  eventSource.onopen = () => {
    connected.value = true
  }

  eventSource.onmessage = (e) => {
    const data = JSON.parse(e.data)
    notifications.value.unshift(data)

    // 最多保留 50 条
    if (notifications.value.length > 50) {
      notifications.value.pop()
    }
  }

  eventSource.addEventListener('alert', (e) => {
    const alert = JSON.parse(e.data)
    // 显示弹窗通知
    showToast(alert.message)
  })

  eventSource.onerror = () => {
    connected.value = false
    // 浏览器默认会自动重连
  }
}

function disconnect() {
  eventSource?.close()
  eventSource = null
  connected.value = false
}

onUnmounted(disconnect)
</script>

<template>
  <div class="notification-center">
    <span :class="{ online: connected }">{{ connected ? '已连接' : '未连接' }}</span>
    <button @click="connected ? disconnect() : connect(token)">
      {{ connected ? '断开' : '连接' }}
    </button>
    <ul>
      <li v-for="(n, i) in notifications" :key="i">
        {{ n.title }}: {{ n.content }}
      </li>
    </ul>
  </div>
</template>
```

## 五、SSE vs WebSocket 对比

| 特性 | SSE | WebSocket |
|------|-----|-----------|
| 方向 | 服务器→客户端（单向） | 双向通信 |
| 协议 | HTTP | WS/WSS |
| 自动重连 | 是 | 需自己实现 |
| 认证 | 需变通 | 支持 header |
| 数据格式 | 文本 | 文本/二进制 |
| 浏览器限制 | 同域6个连接 | 无限制 |
| 适用场景 | 通知、数据更新 | 聊天、游戏 |

## 六、fetch 实现 SSE（支持自定义 header）

```js
async function sseWithAuth(url, token, onMessage) {
  const res = await fetch(url, {
    headers: { Authorization: `Bearer ${token}` }
  })

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop()  // 保留不完整的行

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        onMessage(JSON.parse(line.slice(6)))
      }
    }
  }
}
```

## 三、注意事项与常见陷阱

- SSE 默认自动重连，但 `EventSource` 不支持自定义 header
- 需要认证时可使用 URL 参数传递 token，或用 `fetch` API 手动实现
- 浏览器对同一域名的 SSE 连接数有限制（通常 6 个）
- 组件卸载时务必调用 `eventSource.close()`
- SSE 只支持文本数据，二进制数据需用 WebSocket
