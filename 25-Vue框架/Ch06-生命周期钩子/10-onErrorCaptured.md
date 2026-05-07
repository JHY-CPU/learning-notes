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
  console.error('错误来源组件:', instance)
  console.error('错误信息:', info)

  hasError.value = true
  errorMsg.value = err.message

  // 返回 false 阻止错误继续传播
  return false
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
  return false  // 阻止错误传播
})

function retry() { error.value = null }
</script>
```

## 三、注意事项与常见陷阱
- 返回 `false` 阻止错误传播，返回 `true` 或不返回则继续传播
- `info` 参数说明错误发生在哪个生命周期（如 `render`、`mounted`）
- 全局错误可以用 `app.config.errorHandler` 处理
- 错误边界组件应放在可能出错的组件的父级
