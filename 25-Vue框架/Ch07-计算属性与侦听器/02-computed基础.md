# computed 基础

## 一、概念说明
`computed` 是一个**有缓存的计算属性**。它根据依赖的响应式数据自动计算值，只有依赖变化时才重新计算。相比 methods，computed 具有缓存优势。

## 二、具体用法

### 2.1 基本 getter
```vue
<script setup>
import { ref, computed } from 'vue'

const firstName = ref('张')
const lastName = ref('三')

// 只读计算属性
const fullName = computed(() => {
  return `${firstName.value} ${lastName.value}`
})
</script>

<template>
  <p>全名: {{ fullName }}</p>
</template>
```

### 2.2 自动依赖追踪
```vue
<script setup>
import { ref, computed } from 'vue'

const items = ref([
  { name: '苹果', price: 5, count: 3 },
  { name: '香蕉', price: 3, count: 5 }
])

// computed 自动追踪 items 的变化
const totalPrice = computed(() => {
  return items.value.reduce((sum, item) => sum + item.price * item.count, 0)
})
</script>
```

### 2.3 TypeScript 类型
```vue
<script setup>
import { ref, computed } from 'vue'

const count = ref(0)
// TypeScript 自动推断类型为 ComputedRef<number>
const doubled = computed(() => count.value * 2)
</script>
```

## 三、注意事项与常见陷阱
- computed 返回的是 `ComputedRef`，读取值需要 `.value`（模板中自动解包）
- 有**缓存**：依赖不变时，多次访问不会重新计算
- 不要在 computed 中执行副作用（如修改数据、发请求）
- computed 的依赖必须是响应式的，普通变量不会触发更新
