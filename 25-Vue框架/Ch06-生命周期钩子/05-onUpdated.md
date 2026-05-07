# onUpdated

## 一、概念说明
`onUpdated` 在组件**响应式数据变化导致 DOM 重新渲染之后**调用。此时 DOM 已经更新完毕，可以安全地访问更新后的 DOM 状态。

## 二、具体用法

### 2.1 访问更新后的 DOM
```vue
<template>
  <ul ref="listRef">
    <li v-for="item in items" :key="item.id">{{ item.name }}</li>
  </ul>
</template>
<script setup>
import { ref, onUpdated } from 'vue'

const items = ref([{ id: 1, name: '张三' }])
const listRef = ref(null)

onUpdated(() => {
  // DOM 已更新
  console.log('列表项数:', listRef.value.children.length)
})
</script>
```

### 2.2 滚动到底部（聊天场景）
```vue
<template>
  <div ref="chatRef" class="chat">
    <p v-for="msg in messages" :key="msg.id">{{ msg.text }}</p>
  </div>
</template>
<script setup>
import { ref, onUpdated } from 'vue'

const messages = ref([])
const chatRef = ref(null)

onUpdated(() => {
  // 新消息时自动滚动到底部
  chatRef.value.scrollTop = chatRef.value.scrollHeight
})
</script>
```

## 三、注意事项与常见陷阱
- **不要在 `onUpdated` 中修改响应式数据**，会导致无限循环
- 子组件的 `onUpdated` 先于父组件执行
- 此钩子在服务端渲染中不会被调用
- 如果需要在更新后执行操作，优先使用 `watch` + `nextTick`
