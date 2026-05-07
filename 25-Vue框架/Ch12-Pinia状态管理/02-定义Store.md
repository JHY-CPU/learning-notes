# 定义Store

## 一、概念说明

`defineStore()`是定义Store的核心函数。它接收store的唯一ID和配置对象（或setup函数），返回一个使用该store的函数。

```js
// stores/counter.js
import { defineStore } from 'pinia'

// 方式1：选项式API风格
export const useCounter = defineStore('counter', {
  state: () => ({ count: 0 }),
  getters: { doubled: (state) => state.count * 2 },
  actions: {
    increment() { this.count++ }
  }
})

// 方式2：Setup函数风格（推荐）
export const useCounter = defineStore('counter', () => {
  const count = ref(0)
  const doubled = computed(() => count.value * 2)
  const increment = () => count.value++
  return { count, doubled, increment }
})
```

## 二、具体用法

### 在组件中使用

```vue
<script setup>
import { useCounter } from '@/stores/counter'

const counter = useCounter()

// 直接访问
console.log(counter.count)

// 调用action
counter.increment()
</script>

<template>
  <p>{{ counter.count }}</p>
  <button @click="counter.increment">+1</button>
</template>
```

### Store ID的作用

- 用于DevTools识别
- 支持服务端渲染
- 必须全局唯一

## 三、注意事项与常见陷阱

1. Store ID必须唯一，建议与文件名一致
2. Setup风格更适合TypeScript，更灵活
3. 不要在组件外直接调用useStore（需在setup中）
4. Store是单例，多次调用useStore返回同一实例
5. 可以在Pinia插件中访问store.$id获取ID
