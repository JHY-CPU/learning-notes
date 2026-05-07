# WebSocket 集成

## 一、概念说明

WebSocket 提供浏览器与服务器之间的**全双工**实时通信。与 SSE 的单向推送不同，WebSocket 允许双方随时发送消息，适合聊天、协作编辑、实时游戏等场景。

```vue
<script setup>
import { ref, onUnmounted } from 'vue'

const messages = ref([])
const input = ref('')
let ws = null

function connect() {
  ws = new WebSocket('ws://localhost:3000/ws')

  ws.onopen = () => console.log('已连接')
  ws.onmessage = (e) => {
    messages.value.push(JSON.parse(e.data))
  }
  ws.onclose = () => console.log('已断开')
}

function send() {
  if (ws && input.value) {
    ws.send(JSON.stringify({ type: 'chat', content: input.value }))
    input.value = ''
  }
}

onUnmounted(() => ws?.close())
</script>

<template>
  <button @click="connect">连接</button>
  <div v-for="(msg, i) in messages" :key="i">{{ msg.content }}</div>
  <input v-model="input" @keyup.enter="send" placeholder="输入消息" />
  <button @click="send">发送</button>
</template>
```

## 二、具体用法

### 2.1 封装 Composable

```js
// composables/useWebSocket.js
import { ref, onUnmounted } from 'vue'

export function useWebSocket(url) {
  const data = ref(null)
  const status = ref('disconnected')
  let ws = null

  function connect() {
    ws = new WebSocket(url)
    status.value = 'connecting'
    ws.onopen = () => { status.value = 'connected' }
    ws.onmessage = (e) => { data.value = JSON.parse(e.data) }
    ws.onclose = () => { status.value = 'disconnected' }
  }

  function send(payload) {
    ws?.send(JSON.stringify(payload))
  }

  function close() {
    ws?.close()
  }

  onUnmounted(close)
  return { data, status, connect, send, close }
}
```

### 2.2 自动重连

```js
function connectWithRetry(url, maxRetries = 5) {
  let retries = 0
  function attempt() {
    const ws = new WebSocket(url)
    ws.onclose = () => {
      if (retries < maxRetries) {
        retries++
        setTimeout(attempt, 1000 * retries)
      }
    }
  }
  attempt()
}
```

## 三、注意事项与常见陷阱

- 组件卸载时必须关闭 WebSocket 连接
- 需要处理心跳机制，防止连接被服务器超时断开
- 二进制数据需设置 `ws.binaryType = 'arraybuffer'`
