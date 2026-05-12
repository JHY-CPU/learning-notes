# computed 链式调用

## 一、概念说明
计算属性可以**依赖其他计算属性**，形成链式依赖。Vue 会正确追踪整条依赖链，确保在最基础的响应式数据变化时，所有依赖的 computed 都会更新。

## 二、具体用法

### 2.1 链式依赖
```vue
<script setup>
import { ref, computed } from 'vue'

const price = ref(100)
const quantity = ref(3)

// 第一层：小计
const subtotal = computed(() => price.value * quantity.value)

// 第二层：依赖 subtotal
const tax = computed(() => subtotal.value * 0.1)

// 第三层：依赖 subtotal 和 tax
const total = computed(() => subtotal.value + tax.value)
</script>

<template>
  <p>单价: {{ price }}，数量: {{ quantity }}</p>
  <p>小计: {{ subtotal }}</p>
  <p>税额: {{ tax }}</p>
  <p>总计: {{ total }}</p>
</template>
<!-- 修改 price → subtotal 变化 → tax 变化 → total 变化 -->
```

### 2.2 基于 computed 过滤/排序
```vue
<script setup>
import { ref, computed } from 'vue'

const users = ref([
  { name: '张三', age: 25, active: true },
  { name: '李四', age: 30, active: false }
])

const activeUsers = computed(() => users.value.filter(u => u.active))
const sortedActiveUsers = computed(() =>
  [...activeUsers.value].sort((a, b) => a.age - b.age)
)
</script>
```

### 2.3 多数据源聚合
```vue
<script setup>
import { ref, computed } from 'vue'

const cartItems = ref([])
const discount = ref(0)
const shipping = ref(10)

const cartTotal = computed(() =>
  cartItems.value.reduce((sum, i) => sum + i.price * i.qty, 0)
)
const finalPrice = computed(() =>
  Math.max(0, cartTotal.value - discount.value + shipping.value)
)
</script>
```

## 三、注意事项与常见陷阱
- 链式 computed 的更新是**同步批量**的，不会中间状态不一致
- 避免过长的依赖链（超过 3-4 层），考虑简化数据结构
- 循环依赖会导致栈溢出（A 依赖 B，B 依赖 A）
- 链中的每个 computed 都有独立的缓存

## 四、性能优化技巧

### 4.1 避免不必要的中间计算
```vue
<script setup>
import { ref, computed } from 'vue'

const items = ref([])

// ❌ 链过长，每层都有缓存开销
const filtered = computed(() => items.value.filter(i => i.active))
const sorted = computed(() => [...filtered.value].sort((a, b) => a.price - b.price))
const formatted = computed(() => sorted.value.map(i => ({ ...i, label: `${i.name} ¥${i.price}` })))

// ✅ 适当合并减少层数
const processedItems = computed(() =>
  items.value
    .filter(i => i.active)
    .sort((a, b) => a.price - b.price)
    .map(i => ({ ...i, label: `${i.name} ¥${i.price}` }))
)
</script>
```

### 4.2 大数据量时避免频繁重新排序
```vue
<script setup>
import { ref, computed } from 'vue'

const products = ref([])
const sortField = ref('name')
const sortOrder = ref('asc')

// 缓存排序结果，只在排序条件变化时重新排序
const sortedProducts = computed(() => {
  const sorted = [...products.value]
  sorted.sort((a, b) => {
    const modifier = sortOrder.value === 'asc' ? 1 : -1
    return a[sortField.value] > b[sortField.value] ? modifier : -modifier
  })
  return sorted
})

const paginatedProducts = computed(() => {
  const page = 1
  const size = 20
  return sortedProducts.value.slice((page - 1) * size, page * size)
})
</script>
```

## 五、调试依赖链

```js
import { ref, computed, effect } from 'vue'

const base = ref(1)
const step1 = computed(() => {
  console.log('[step1] 计算')
  return base.value * 2
})
const step2 = computed(() => {
  console.log('[step2] 计算')
  return step1.value + 10
})
const step3 = computed(() => {
  console.log('[step3] 计算')
  return step2.value / 3
})

// 修改 base 时，输出：
// [step1] 计算
// [step2] 计算
// [step3] 计算
// 链中的所有 computed 都会按顺序更新
```
