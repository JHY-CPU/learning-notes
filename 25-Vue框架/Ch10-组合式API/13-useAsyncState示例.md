# useAsyncState示例

## 一、概念说明

`useAsyncState`管理异步操作的状态，统一处理loading、error、data三个状态。适合任何需要异步操作的场景。

```vue
<template>
  <div>
    <p>状态: {{ isReady ? '就绪' : isLoading ? '加载中' : '未开始' }}</p>
    <p v-if="error">错误: {{ error.message }}</p>
    <p v-if="state">结果: {{ state }}</p>
    <button @click="execute(1000)">延迟加载</button>
  </div>
</template>

<script setup>
import { useAsyncState } from './composables/useAsyncState'

const delay = (ms) => new Promise(r => setTimeout(r, ms))

const { state, isReady, isLoading, error, execute } = useAsyncState(
  async (ms) => {
    await delay(ms)
    return { message: `延迟${ms}ms完成` }
  },
  null,  // 初始值
  { immediate: false }
)
</script>
```

## 二、具体用法

### 完整实现

```js
// composables/useAsyncState.js
import { ref, shallowRef } from 'vue'

export function useAsyncState(promise, initialState, options = {}) {
  const {
    immediate = true,
    delay = 0,
    onSuccess,
    onError,
    resetOnExecute = false
  } = options

  const state = shallowRef(initialState)
  const isReady = ref(false)
  const isLoading = ref(false)
  const error = ref(undefined)

  const execute = async (...args) => {
    if (resetOnExecute) state.value = initialState
    isLoading.value = true
    isReady.value = false
    error.value = undefined

    if (delay > 0) {
      await new Promise(r => setTimeout(r, delay))
    }

    try {
      state.value = await promise(...args)
      isReady.value = true
      onSuccess?.(state.value)
    } catch (e) {
      error.value = e
      onError?.(e)
    } finally {
      isLoading.value = false
    }

    return state.value
  }

  if (immediate) execute()

  return { state, isReady, isLoading, error, execute }
}
```

## 三、注意事项与常见陷阱

1. 使用`shallowRef`存储大数据对象，避免深度响应式带来的性能开销
2. 异步函数的参数通过`execute`传递
3. 注意取消未完成的异步操作，避免内存泄漏
4. `error`是ref，可能是`Error`实例或`undefined`
5. 多次快速调用`execute`可能导致竞态条件，考虑加入取消机制
