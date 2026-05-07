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
