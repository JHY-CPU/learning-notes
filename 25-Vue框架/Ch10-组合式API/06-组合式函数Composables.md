# 组合式函数Composables

## 一、概念说明

**组合式函数 (Composables)** 是利用Vue组合式API封装和复用有状态逻辑的函数。它类似于React中的自定义Hooks，以`use`前缀命名。

```vue
<template>
  <p>鼠标位置: {{ x }}, {{ y }}</p>
</template>

<script setup>
import { useMouse } from './composables/useMouse'

const { x, y } = useMouse()
</script>
```

```js
// composables/useMouse.js
import { ref, onMounted, onUnmounted } from 'vue'

export function useMouse() {
  const x = ref(0)
  const y = ref(0)

  function update(e) {
    x.value = e.pageX
    y.value = e.pageY
  }

  onMounted(() => window.addEventListener('mousemove', update))
  onUnmounted(() => window.removeEventListener('mousemove', update))

  return { x, y }
}
```

## 二、具体用法

### 定义组合式函数的规范

```js
// composables/useCounter.js
import { ref, computed } from 'vue'

export function useCounter(initialValue = 0) {
  const count = ref(initialValue)
  const doubled = computed(() => count.value * 2)

  const increment = () => count.value++
  const decrement = () => count.value--
  const reset = () => { count.value = initialValue }

  return { count, doubled, increment, decrement, reset }
}
```

### 在组件中使用

```vue
<script setup>
import { useCounter } from './composables/useCounter'

const { count, doubled, increment, reset } = useCounter(10)
</script>
```

## 三、注意事项与常见陷阱

1. **命名规范**：文件名和函数名都以`use`开头（如`useMouse.js`、`useMouse()`）
2. 组合式函数可以在**其他**组合式函数中调用
3. 每次调用组合式函数都会创建**独立**的状态实例
4. 返回值推荐使用`ref`而非`reactive`，避免解构丢失响应式
5. 组合式函数**不是组件**，不包含模板
6. 确保生命周期钩子在正确的上下文中（setup执行时）调用

## 四、实用 Composable 示例

### 4.1 本地存储
```js
// composables/useLocalStorage.js
import { ref, watch } from 'vue'

export function useLocalStorage(key, defaultValue) {
  const stored = localStorage.getItem(key)
  const data = ref(stored ? JSON.parse(stored) : defaultValue)

  watch(data, (val) => {
    localStorage.setItem(key, JSON.stringify(val))
  }, { deep: true })

  return data
}
```

### 4.2 异步数据获取
```js
// composables/useFetch.js
import { ref, watchEffect } from 'vue'

export function useFetch(url) {
  const data = ref(null)
  const error = ref(null)
  const loading = ref(false)

  watchEffect(async (onInvalidate) => {
    let cancelled = false
    onInvalidate(() => { cancelled = true })

    loading.value = true
    error.value = null

    try {
      const res = await fetch(url.value || url)
      if (!cancelled) {
        data.value = await res.json()
      }
    } catch (e) {
      if (!cancelled) error.value = e
    } finally {
      if (!cancelled) loading.value = false
    }
  })

  return { data, error, loading }
}
```

### 4.3 防抖值
```js
// composables/useDebounce.js
import { ref, watch } from 'vue'

export function useDebounce(value, delay = 300) {
  const debouncedValue = ref(value.value)
  let timer = null

  watch(value, (newVal) => {
    clearTimeout(timer)
    timer = setTimeout(() => {
      debouncedValue.value = newVal
    }, delay)
  })

  return debouncedValue
}
```

## 五、Composable 的组织规范

```
src/
  composables/
    useMouse.js         # 鼠标位置追踪
    useFetch.js         # 数据获取
    useLocalStorage.js  # 本地存储
    useDebounce.js      # 防抖
    useAuth.js          # 认证逻辑
    index.js            # 统一导出
```

```js
// composables/index.js
export { useMouse } from './useMouse'
export { useFetch } from './useFetch'
export { useLocalStorage } from './useLocalStorage'
```

## 六、Composable vs Mixins 对比

| 特性 | Composables | Mixins |
|------|-------------|--------|
| 命名冲突 | 无（函数作用域） | 有（合并冲突） |
| 数据来源 | 清晰（返回值） | 隐式（混入） |
| TypeScript | 良好支持 | 支持差 |
| 可组合性 | 高（互相调用） | 低 |
| 代码可读性 | 高 | 低 |
