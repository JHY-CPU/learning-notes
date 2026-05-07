# onBeforeUpdate

## 一、概念说明
`onBeforeUpdate` 在组件**响应式数据变化导致 DOM 重新渲染之前**调用。此时可以访问更新前的 DOM 状态，适合做 DOM 更新前的准备工作。

## 二、具体用法

### 2.1 获取更新前的 DOM 状态
```vue
<template>
  <div ref="listRef">
    <p v-for="item in items" :key="item.id">{{ item.text }}</p>
  </div>
  <button @click="addItem">添加</button>
</template>
<script setup>
import { ref, onBeforeUpdate, onUpdated } from 'vue'

const items = ref([{ id: 1, text: '第一项' }])
const listRef = ref(null)

onBeforeUpdate(() => {
  // 此时 DOM 还未更新，仍是旧状态
  console.log('更新前子节点数:', listRef.value?.children.length)
})

onUpdated(() => {
  console.log('更新后子节点数:', listRef.value?.children.length)
})

function addItem() {
  items.value.push({ id: Date.now(), text: '新项' })
}
</script>
```

### 2.2 保存滚动位置
```vue
<script setup>
import { ref, onBeforeUpdate } from 'vue'

const scrollTop = ref(0)
const containerRef = ref(null)

onBeforeUpdate(() => {
  // 保存更新前的滚动位置
  scrollTop.value = containerRef.value?.scrollTop || 0
})
</script>
```

## 三、注意事项与常见陷阱
- **不要在 `onBeforeUpdate` 中修改响应式数据**，会导致无限循环
- 子组件的 `onBeforeUpdate` 先于父组件执行
- 此钩子在服务端渲染中不会被调用
- 主要用于获取更新前的 DOM 快照，而非修改状态
