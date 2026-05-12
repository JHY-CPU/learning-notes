# Pinia 跨组件通信

## 一、概念说明
Pinia 是 Vue 官方推荐的**状态管理库**，作为组件通信的中枢，任何组件都可以读取和修改 store 中的状态。相比事件总线，Pinia 提供了更好的 DevTools 支持和类型安全。

## 二、具体用法

### 2.1 定义 Store
```js
// stores/message.js
import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useMessageStore = defineStore('message', () => {
  const messages = ref([])
  const unread = ref(0)

  function addMessage(msg) {
    messages.value.push(msg)
    unread.value++
  }

  function clearUnread() {
    unread.value = 0
  }

  return { messages, unread, addMessage, clearUnread }
})
```

### 2.2 组件 A：写入 Store
```vue
<script setup>
import { useMessageStore } from '@/stores/message'
import { ref } from 'vue'

const store = useMessageStore()
const text = ref('')

function send() {
  store.addMessage({ text: text.value, time: Date.now() })
  text.value = ''
}
</script>
```

### 2.3 组件 B：读取 Store
```vue
<script setup>
import { useMessageStore } from '@/stores/message'
const store = useMessageStore()
</script>

<template>
  <p>未读消息: {{ store.unread }}</p>
  <ul>
    <li v-for="msg in store.messages" :key="msg.time">
      {{ msg.text }}
    </li>
  </ul>
</template>
```

### 2.4 订阅状态变化
```js
store.$subscribe((mutation, state) => {
  console.log('状态变化:', mutation.type, state)
  // mutation.type: 'direct' | 'patch object' | 'patch function'
})
```

### 2.5 storeToRefs 解构
```vue
<script setup>
import { storeToRefs } from 'pinia'
import { useMessageStore } from '@/stores/message'

const store = useMessageStore()
// 解构保持响应式
const { messages, unread } = storeToRefs(store)
// 方法可以直接解构
const { addMessage, clearUnread } = store
</script>
```

## 三、常见用例

| 场景 | Store 设计 |
|------|-----------|
| 用户认证 | `useAuthStore` |
| 购物车 | `useCartStore` |
| 消息通知 | `useNotificationStore` |
| 全局配置 | `useConfigStore` |

## 四、注意事项与常见陷阱

- Pinia store 中的 ref 会被自动解包，不需要 `.value`
- 适合跨多个组件共享的状态，局部状态仍用 ref/reactive
- `$subscribe` 在组件卸载时自动取消，无需手动清理
- 避免在 store 中存储 UI 状态（如弹窗显隐），保持 store 专注于业务数据
- 使用 `storeToRefs` 解构 store 避免丢失响应式
- Pinia 支持多个 store 之间互相引用
