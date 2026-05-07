# 组合式 API 概览

## 一、概念说明

组合式 API（Composition API）是 Vue 3 引入的新编程范式。通过 `setup()` 函数和 `ref`、`reactive`、`computed`、`watch` 等函数，将相关逻辑组织在一起，解决了选项式 API 在大型组件中逻辑分散的问题。

`<script setup>` 是 Composition API 的语法糖，省略了 `setup()` 函数的样板代码，是 Vue 3 推荐的主要写法。

```vue
<script setup>
import { ref, computed, onMounted } from 'vue'

const count = ref(0)
const doubled = computed(() => count.value * 2)

function increment() {
  count.value++
}

onMounted(() => {
  console.log('组件已挂载')
})
</script>

<template>
  <p>{{ count }} × 2 = {{ doubled }}</p>
  <button @click="increment">+1</button>
</template>
```

## 二、具体用法

### 2.1 响应式基础

```vue
<script setup>
import { ref, reactive } from 'vue'

// ref：基本类型响应式
const count = ref(0)
console.log(count.value) // 访问需要 .value

// reactive：对象响应式
const state = reactive({
  name: '张三',
  age: 25
})
console.log(state.name) // 直接访问，不需要 .value
</script>
```

### 2.2 生命周期钩子

```vue
<script setup>
import {
  onBeforeMount, onMounted,
  onBeforeUpdate, onUpdated,
  onBeforeUnmount, onUnmounted
} from 'vue'

onMounted(() => {
  // DOM 已挂载，可以安全操作 DOM
  console.log('mounted')
})

onUnmounted(() => {
  // 清理定时器、事件监听等
  console.log('unmounted')
})
</script>
```

### 2.3 组合式函数（Composables）

```js
// composables/useCounter.js
import { ref, computed } from 'vue'

export function useCounter(initialValue = 0) {
  const count = ref(initialValue)
  const doubled = computed(() => count.value * 2)
  const increment = () => count.value++
  const reset = () => { count.value = initialValue }

  return { count, doubled, increment, reset }
}
```

```vue
<script setup>
import { useCounter } from '@/composables/useCounter'

const { count, doubled, increment } = useCounter(10)
</script>

<template>
  <p>{{ count }} → {{ doubled }}</p>
  <button @click="increment">+1</button>
</template>
```

## 三、注意事项与常见陷阱

- `ref` 在模板中自动解包，不需要 `.value`；在 JS 中需要
- `reactive` 解构会丢失响应式，使用 `toRefs()` 保持
- 不要在 `setup()` 中使用 `this`，它尚未创建
- 组合式函数名以 `use` 开头，遵循命名约定
- `onMounted` 中才能安全访问 DOM 元素
