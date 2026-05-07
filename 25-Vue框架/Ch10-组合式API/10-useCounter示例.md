# useCounter示例

## 一、概念说明

`useCounter`是一个经典的组合式函数示例，封装了计数器的所有逻辑，包括状态、计算属性和操作方法。它展示了组合式函数的基本结构和用法。

```vue
<template>
  <div>
    <p>当前值: {{ count }}</p>
    <p>双倍值: {{ doubled }}</p>
    <p>是否偶数: {{ isEven ? '是' : '否' }}</p>
    <button @click="increment">+1</button>
    <button @click="decrement">-1</button>
    <button @click="reset">重置</button>
  </div>
</template>

<script setup>
import { useCounter } from './composables/useCounter'

const { count, doubled, isEven, increment, decrement, reset } = useCounter(0)
</script>
```

## 二、具体用法

### 完整的useCounter实现

```js
// composables/useCounter.js
import { ref, computed } from 'vue'

export function useCounter(initialValue = 0, options = {}) {
  const { min = -Infinity, max = Infinity, step = 1 } = options

  const count = ref(initialValue)
  const doubled = computed(() => count.value * 2)
  const isEven = computed(() => count.value % 2 === 0)

  const increment = () => {
    if (count.value + step <= max) {
      count.value += step
    }
  }

  const decrement = () => {
    if (count.value - step >= min) {
      count.value -= step
    }
  }

  const reset = () => {
    count.value = initialValue
  }

  const set = (val) => {
    if (val >= min && val <= max) {
      count.value = val
    }
  }

  return {
    count, doubled, isEven,
    increment, decrement, reset, set
  }
}
```

### 带范围限制的计数器

```vue
<script setup>
import { useCounter } from './composables/useCounter'

const { count, increment, decrement } = useCounter(5, {
  min: 0,
  max: 10,
  step: 1
})
</script>
```

## 三、注意事项与常见陷阱

1. `useCounter`返回的`count`是`ref`，在JS中需要`.value`访问
2. 每次调用`useCounter`创建独立的计数器实例
3. 可以在一个组件中使用多个`useCounter`，各自独立
4. 参数通过options对象传递，保持API灵活
5. 返回的方法已经绑定了对应的ref，可直接作为事件处理函数
