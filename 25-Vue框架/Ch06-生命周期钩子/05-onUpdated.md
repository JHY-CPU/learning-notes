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

## 四、与 nextTick 的对比

```vue
<script setup>
import { ref, onUpdated, nextTick, watch } from 'vue'

const items = ref([])

// 方式 1：onUpdated（每次更新都执行）
onUpdated(() => {
  console.log('DOM 更新了')
})

// 方式 2：watch + nextTick（只有 items 变化时执行）
watch(items, async () => {
  await nextTick()
  console.log('items 更新后的 DOM')
})
</script>
```

## 五、实际使用场景

| 场景 | 说明 |
| --- | --- |
| 聊天自动滚动 | 新消息后滚动到底部 |
| 列表高度测量 | 列表变化后重新计算高度 |
| 第三方库刷新 | 数据变化后通知库重新渲染 |
| 动画触发 | 列表增删后的入场/出场动画 |
