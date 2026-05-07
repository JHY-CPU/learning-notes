# Web Workers

## 一、概念说明

Web Worker 在独立线程中运行 JavaScript，不阻塞主线程（UI 渲染）。适合处理大数据计算、复杂数学运算、图像处理等密集型任务。

```vue
<script setup>
import { ref } from 'vue'

const result = ref(null)
const computing = ref(false)

function calculate() {
  computing.value = true

  // 创建 Worker
  const worker = new Worker(
    new URL('../workers/fibonacci.js', import.meta.url),
    { type: 'module' }
  )

  worker.postMessage({ n: 45 })

  worker.onmessage = (e) => {
    result.value = e.data
    computing.value = false
    worker.terminate()
  }
}
</script>

<template>
  <button @click="calculate" :disabled="computing">
    {{ computing ? '计算中...' : '计算斐波那契' }}
  </button>
  <p v-if="result !== null">结果: {{ result }}</p>
</template>
```

```js
// workers/fibonacci.js
self.onmessage = (e) => {
  const { n } = e.data
  const result = fibonacci(n)
  self.postMessage(result)
}

function fibonacci(n) {
  if (n <= 1) return n
  return fibonacci(n - 1) + fibonacci(n - 2)
}
```

## 二、具体用法

### 2.1 Worker 文件要求

```js
// workers/worker.js
// Worker 中不能访问 DOM、window、document
// 可以使用 fetch、WebSocket、定时器等

self.onmessage = (e) => {
  const data = e.data
  // 处理数据...
  self.postMessage(result)
}
```

### 2.2 封装 Composable

```js
// composables/useWorker.js
import { ref, onUnmounted } from 'vue'

export function useWorker(workerUrl) {
  const result = ref(null)
  const loading = ref(false)
  const worker = new Worker(workerUrl, { type: 'module' })

  worker.onmessage = (e) => {
    result.value = e.data
    loading.value = false
  }

  function post(data) {
    loading.value = true
    worker.postMessage(data)
  }

  onUnmounted(() => worker.terminate())

  return { result, loading, post }
}
```

### 2.3 Worker 中使用 ES Module

```js
// vite 支持 type: 'module' 的 Worker
const worker = new Worker(
  new URL('./worker.js', import.meta.url),
  { type: 'module' }
)
```

## 三、注意事项与常见陷阱

- Worker 中无法访问 DOM，数据需要通过 `postMessage` 传递
- 序列化大对象有性能开销，考虑使用 `Transferable` 对象
- 记得在不用时调用 `worker.terminate()` 释放资源
- Worker 文件路径在构建时需要正确处理
