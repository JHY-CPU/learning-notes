# watchEffect 停止

## 一、概念说明
`watchEffect` 返回一个**停止函数**，调用它可以手动停止侦听。组件卸载时会自动停止所有 watchEffect，但有时需要在组件卸载前手动停止。

## 二、具体用法

### 2.1 手动停止
```vue
<script setup>
import { ref, watchEffect } from 'vue'

const count = ref(0)

const stop = watchEffect(() => {
  console.log('count:', count.value)
})

// 手动停止
function stopWatching() {
  stop()
  console.log('已停止侦听')
}
</script>

<template>
  <button @click="count++">+1</button>
  <button @click="stopWatching">停止侦听</button>
</template>
```

### 2.2 条件停止
```vue
<script setup>
import { ref, watchEffect } from 'vue'

const score = ref(0)
const maxScore = ref(100)

const stop = watchEffect(() => {
  console.log('分数:', score.value)
  if (score.value >= maxScore.value) {
    console.log('达到最大值，停止侦听')
    stop()
  }
})
</script>
```

### 2.3 组件卸载时自动停止
```vue
<script setup>
import { ref, watchEffect, onUnmounted } from 'vue'

const data = ref(0)

// 组件卸载时自动停止，无需手动处理
watchEffect(() => {
  console.log(data.value)
})

// 如果需要在卸载前做额外处理
onUnmounted(() => {
  console.log('组件卸载，watchEffect 已自动停止')
})
</script>
```

### 2.4 在 Composable 中使用
```js
export function usePolling(callback, interval) {
  const stop = watchEffect((onInvalidate) => {
    const timer = setInterval(callback, interval.value)
    onInvalidate(() => clearInterval(timer))
  })

  // 返回停止函数供调用者使用
  return { stop }
}
```

## 三、注意事项与常见陷阱
- 停止后会调用清理函数（onInvalidate 注册的回调）
- 组件卸载时自动停止，但 Composable 应暴露停止函数
- 同一个 stop 函数多次调用是安全的（幂等）
- 停止后无法重新启动，需要重新创建 watchEffect
