# computed 与 methods 区别

## 一、概念说明
`computed` 和 `methods` 都可以返回派生数据，但核心区别在于**缓存**。computed 会缓存计算结果，只有依赖变化时才重新计算；methods 每次调用都会重新执行。

## 二、具体用法

### 2.1 缓存 vs 无缓存
```vue
<script setup>
import { ref, computed } from 'vue'

const items = ref([1, 2, 3, 4, 5])

// computed: 有缓存，items 不变时返回缓存值
const total = computed(() => {
  console.log('computed 计算中...')  // 只在 items 变化时打印
  return items.value.reduce((a, b) => a + b, 0)
})

// methods: 无缓存，每次调用都执行
function getTotal() {
  console.log('method 计算中...')  // 每次调用都打印
  return items.value.reduce((a, b) => a + b, 0)
}
</script>

<template>
  <!-- computed 在模板中多次使用只计算一次 -->
  <p>{{ total }}</p>
  <p>{{ total }}</p>
  <p>{{ total }}</p>

  <!-- method 每次调用都计算 -->
  <p>{{ getTotal() }}</p>
  <p>{{ getTotal() }}</p>
  <p>{{ getTotal() }}</p>
</template>
```

### 2.2 性能对比
```vue
<!-- 当 computed 依赖不变时，即使其他数据变化触发重新渲染，
     computed 也不会重新计算，直接返回缓存值 -->
<script setup>
import { ref, computed } from 'vue'

const expensive = ref([/* 大数组 */])
const unrelated = ref(0)

const result = computed(() => {
  // 昂贵的计算
  return expensive.value.reduce(/* 复杂逻辑 */)
})
// 修改 unrelated 不会触发 result 重新计算
</script>
```

### 2.3 何时用 methods
```vue
<script setup>
// 需要传参时 → 用 methods
function filterBy(keyword) {
  return items.value.filter(i => i.includes(keyword))
}

// 不需要传参，依赖响应式数据 → 用 computed
const filtered = computed(() => items.value.filter(i => i.includes('Vue')))
</script>
```

## 三、注意事项与常见陷阱
- computed 适合**不需要参数**的纯派生计算
- methods 适合**需要传参**或**需要每次执行**的逻辑
- 模板中调用 method 会每次重新执行，注意性能
- 如果计算结果相同且不需参数，优先使用 computed
