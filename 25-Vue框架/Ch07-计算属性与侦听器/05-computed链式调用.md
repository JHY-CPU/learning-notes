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
