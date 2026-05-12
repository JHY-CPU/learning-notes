# onErrorCaptured

## 一、概念说明
`onErrorCaptured` 用于捕获**后代组件**抛出的错误。它可以阻止错误继续向上传播，常用于实现全局错误处理和降级 UI。

## 二、具体用法

### 2.1 基本错误捕获
```vue
<script setup>
import { ref, onErrorCaptured } from 'vue'

const hasError = ref(false)
const errorMsg = ref('')

onErrorCaptured((err, instance, info) => {
  console.error('捕获到错误:', err.message)
  hasError.value = true
  errorMsg.value = err.message
  return false  // 阻止传播
})
</script>

<template>
  <div v-if="hasError">
    <p>出错了: {{ errorMsg }}</p>
    <button @click="hasError = false">重试</button>
  </div>
  <slot v-else />
</template>
```

### 2.2 错误边界组件
```vue
<!-- ErrorBoundary.vue -->
<template>
  <slot v-if="!error" />
  <div v-else class="error-fallback">
    <h3>组件渲染出错</h3>
    <p>{{ error.message }}</p>
    <button @click="retry">重试</button>
  </div>
</template>
<script setup>
import { ref, onErrorCaptured } from 'vue'

const error = ref(null)

onErrorCaptured((err) => {
  error.value = err
  return false
})

function retry() { error.value = null }
</script>
```

### 2.3 错误上报
```vue
<script setup>
import { onErrorCaptured } from 'vue'

onErrorCaptured((err, instance, info) => {
  // 上报到监控平台
  reportError({
    message: err.message,
    stack: err.stack,
    component: instance?.$options?.name || 'Unknown',
    lifecycle: info,
    timestamp: Date.now()
  })
  return false
})
</script>
```

## 三、注意事项与常见陷阱
- 返回 `false` 阻止错误传播，返回 `true` 或不返回则继续传播
- `info` 参数说明错误发生在哪个生命周期（如 `render`、`mounted`）
- 全局错误可以用 `app.config.errorHandler` 处理
- 错误边界组件应放在可能出错的组件的父级

## 四、错误捕获层级

```
app.config.errorHandler（全局）
  -> onErrorCaptured（最内层捕获）
    -> 继续向上传播（除非返回 false）
      -> window.onerror（兜底）
```

## 五、info 参数值

| info 值 | 含义 |
| --- | --- |
| `render` | 渲染过程中出错 |
| `mounted` | 挂载过程中出错 |
| `watcher getter` | 计算属性/watcher 出错 |
| `watcher callback` | watch 回调出错 |
| `setup function` | setup 中出错 |
| `event handler` | 事件处理器出错 |
