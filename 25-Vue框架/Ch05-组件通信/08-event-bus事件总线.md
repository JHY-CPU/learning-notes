# Event Bus 事件总线

## 一、概念说明
事件总线是一种**发布-订阅模式**的通信方式。Vue 3 移除了内置的 `$on`/`$emit`，推荐使用第三方库 `mitt` 实现事件总线。但请注意，事件总线在大型项目中**不推荐使用**，因为它会导致数据流向难以追踪。

## 二、具体用法

### 2.1 安装 mitt
```bash
npm install mitt
```

### 2.2 创建事件总线
```js
// bus.js
import mitt from 'mitt'
export const bus = mitt()
```

### 2.3 组件 A：发送事件
```vue
<script setup>
import { bus } from './bus'

function sendMessage() {
  bus.emit('message', { text: '你好！', from: '组件A' })
}
</script>

<template>
  <button @click="sendMessage">发送消息</button>
</template>
```

### 2.4 组件 B：接收事件
```vue
<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { bus } from './bus'

const messages = ref([])

function handleMessage(data) {
  messages.value.push(data)
}

onMounted(() => {
  bus.on('message', handleMessage)
})
onUnmounted(() => {
  bus.off('message', handleMessage)
})
</script>
```

### 2.5 通配符监听所有事件
```js
bus.on('*', (type, event) => {
  console.log(`事件类型: ${type}`, event)
})
```

## 三、注意事项与常见陷阱
- **务必在 `onUnmounted` 中移除监听**，否则会造成内存泄漏
- 事件总线使组件间产生隐式依赖，大型项目难以维护
- Vue 官方推荐使用 Pinia 替代事件总线进行跨组件通信
- 适合小型项目或临时方案，不建议在生产项目中大量使用
