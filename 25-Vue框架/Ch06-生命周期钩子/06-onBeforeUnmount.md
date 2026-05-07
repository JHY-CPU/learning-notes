# onBeforeUnmount

## 一、概念说明
`onBeforeUnmount` 在组件**卸载之前**调用。此时组件仍然完全可用，是清理副作用（定时器、事件监听、第三方库实例）的最佳时机。

## 二、具体用法

### 2.1 清理定时器
```vue
<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue'

const time = ref(0)
let timer = null

onMounted(() => {
  timer = setInterval(() => {
    time.value++
  }, 1000)
})

onBeforeUnmount(() => {
  // 必须清理，否则内存泄漏
  clearInterval(timer)
  timer = null
})
</script>
```

### 2.2 移除事件监听
```vue
<script setup>
import { onMounted, onBeforeUnmount } from 'vue'

function handleResize() {
  console.log('窗口大小:', window.innerWidth)
}

onMounted(() => {
  window.addEventListener('resize', handleResize)
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', handleResize)
})
</script>
```

### 2.3 销毁第三方库实例
```vue
<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue'
import Chart from 'chart.js'

const canvasRef = ref(null)
let chart = null

onMounted(() => {
  chart = new Chart(canvasRef.value, { type: 'line' })
})

onBeforeUnmount(() => {
  chart?.destroy()
  chart = null
})
</script>
```

## 三、注意事项与常见陷阱
- 所有 `addEventListener`、`setInterval`、`setTimeout` 都应在此清理
- 第三方库创建的实例需要调用其 `destroy`/`dispose` 方法
- WebSocket 连接应在此阶段关闭
- 此钩子中组件仍可正常访问，可以读取最终状态
