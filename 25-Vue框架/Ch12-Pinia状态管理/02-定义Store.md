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

## 四、Setup式 vs 选项式选择

| 特性 | Setup 式 | 选项式 |
|------|---------|--------|
| TypeScript | 自动推断，类型完美 | 需要泛型声明 |
| 灵活性 | 高（可用任何组合式函数） | 中 |
| 学习成本 | 低（类似 Composition API） | 中（类似 Vuex） |
| 代码量 | 较少 | 稍多 |
| 推荐场景 | 新项目、TypeScript | 从 Vuex 迁移 |

## 五、异步初始化

```js
export const useConfig = defineStore('config', () => {
  const config = ref(null)
  const loading = ref(false)

  async function loadConfig() {
    loading.value = true
    try {
      config.value = await api.getConfig()
    } finally {
      loading.value = false
    }
  }

  return { config, loading, loadConfig }
})

// 在 App.vue 中初始化
<script setup>
import { useConfig } from '@/stores/config'
const config = useConfig()
config.loadConfig()
</script>
```

## 三、注意事项与常见陷阱

1. Store ID必须唯一，建议与文件名一致
2. Setup风格更适合TypeScript，更灵活
3. 不要在组件外直接调用useStore（需在setup中）
4. Store是单例，多次调用useStore返回同一实例
5. 可以在Pinia插件中访问store.$id获取ID
6. Setup式Store可以用任何Vue组合式函数（watch、onMounted等）
7. 选项式Store用`this`访问store，Setup式不用
