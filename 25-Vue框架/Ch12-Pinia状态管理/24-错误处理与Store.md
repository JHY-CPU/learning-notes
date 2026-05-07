# 错误处理与Store

## 一、概念说明

在Store中统一处理错误，提供全局错误状态和恢复机制。结合Vue的`onErrorCaptured`实现完整的错误处理链。

```js
export const useAppStore = defineStore('app', () => {
  const globalError = ref(null)

  const setError = (error) => {
    globalError.value = {
      message: error.message || '未知错误',
      code: error.code,
      timestamp: Date.now()
    }
  }

  const clearError = () => { globalError.value = null }

  return { globalError, setError, clearError }
})
```

## 二、具体用法

### Store中的错误处理

```js
export const useOrderStore = defineStore('orders', () => {
  const error = ref(null)

  const createOrder = async (data) => {
    error.value = null
    try {
      return await api.createOrder(data)
    } catch (e) {
      error.value = {
        type: e.response?.status === 401 ? 'auth' : 'api',
        message: e.message
      }
      throw e  // 重新抛出让组件知道
    }
  }

  return { error, createOrder }
})
```

### 全局错误捕获

```js
// App.vue
import { onErrorCaptured } from 'vue'
import { useAppStore } from '@/stores/app'

const appStore = useAppStore()

onErrorCaptured((error) => {
  appStore.setError(error)
  return false  // 阻止继续传播
})
```

### 全局错误展示

```vue
<template>
  <div v-if="appStore.globalError" class="error-banner">
    {{ appStore.globalError.message }}
    <button @click="appStore.clearError">关闭</button>
  </div>
</template>
```

## 三、注意事项与常见陷阱

1. 每个Store可有独立的error状态
2. 全局error用于跨Store的错误展示
3. 错误处理要考虑网络错误、业务错误、权限错误
4. 使用`throw`让调用方知道失败（不要静默吞掉）
5. 提供错误恢复机制（重试、清除错误）
