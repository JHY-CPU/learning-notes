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
  timer = setInterval(() => { time.value++ }, 1000)
})

onBeforeUnmount(() => {
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

### 2.4 关闭 WebSocket 连接
```vue
<script setup>
import { onMounted, onBeforeUnmount } from 'vue'

let ws = null

onMounted(() => {
  ws = new WebSocket('wss://example.com')
  ws.onmessage = (e) => console.log(e.data)
})

onBeforeUnmount(() => {
  if (ws) { ws.close(); ws = null }
})
</script>
```

## 三、注意事项与常见陷阱
- 所有 `addEventListener`、`setInterval`、`setTimeout` 都应在此清理
- 第三方库创建的实例需要调用其 `destroy`/`dispose` 方法
- WebSocket 连接应在此阶段关闭
- 此钩子中组件仍可正常访问，可以读取最终状态
- 使用 AbortController 取消未完成的 fetch 请求

## 四、清理清单

```
onBeforeUnmount 清理清单：
  [ ] clearInterval / clearTimeout
  [ ] removeEventListener
  [ ] ws.close()
  [ ] chart.destroy()
  [ ] abortController.abort()
  [ ] observer.disconnect()  // IntersectionObserver, MutationObserver
  [ ] editor.dispose()       // Monaco Editor, CodeMirror 等
  [ ] map.remove()           // 地图库
```
