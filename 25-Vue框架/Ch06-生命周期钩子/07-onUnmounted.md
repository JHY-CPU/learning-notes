# onUnmounted

## 一、概念说明
`onUnmounted` 在组件**完全卸载之后**调用。此时组件的 DOM 已被移除，所有响应式副作用（computed、watch、effect）已停止。这是组件生命周期的最后一个阶段。

## 二、具体用法

### 2.1 基本用法
```vue
<script setup>
import { onUnmounted } from 'vue'

onUnmounted(() => {
  console.log('组件已完全卸载')
  console.log('DOM 已移除，响应式已停止')
})
</script>
```

### 2.2 最终清理
```vue
<script setup>
import { onMounted, onUnmounted } from 'vue'

let connection = null

onMounted(() => {
  connection = new WebSocket('ws://example.com')
})

onUnmounted(() => {
  // 最终清理
  if (connection) {
    connection.close()
    connection = null
  }
  console.log('WebSocket 连接已关闭')
})
</script>
```

### 2.3 清理全局状态
```vue
<script setup>
import { onUnmounted } from 'vue'
import { useAppStore } from '@/stores/app'

const store = useAppStore()

onUnmounted(() => {
  // 清理组件相关的全局状态
  store.clearComponentData()
})
</script>
```

## 三、注意事项与常见陷阱
- `onUnmounted` 中**不能再访问响应式数据的更新**
- 与 `onBeforeUnmount` 的区别：前者组件仍在，后者已完全销毁
- 此钩子在服务端渲染中不会被调用
- 推荐在 `onBeforeUnmount` 中做清理，`onUnmounted` 做日志记录
