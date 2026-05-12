# watch 选项 flush

## 一、概念说明
`flush` 选项控制 watch 回调的**执行时机**。Vue 的响应式更新是批量异步的，`flush` 决定回调在 DOM 更新之前还是之后执行。

## 二、flush 的三种值

### 2.1 flush: 'pre'（默认）
```vue
<script setup>
import { ref, watch } from 'vue'

const count = ref(0)

// 默认：在 DOM 更新之前执行
watch(count, () => {
  console.log('DOM 更新前触发')
}, { flush: 'pre' })
</script>
```

### 2.2 flush: 'post'
```vue
<template>
  <div ref="el">{{ count }}</div>
</template>
<script setup>
import { ref, watch } from 'vue'

const count = ref(0)
const el = ref(null)

// DOM 更新之后执行，可以访问更新后的 DOM
watch(count, () => {
  console.log('DOM 已更新:', el.value?.textContent)
}, { flush: 'post' })
</script>
```

### 2.3 flush: 'sync'（同步执行）
```vue
<script setup>
import { ref, watch } from 'vue'

const value = ref(0)

// 同步触发，不等待批量更新
watch(value, () => {
  console.log('同步触发')
}, { flush: 'sync' })

value.value = 1  // 立即触发回调
console.log('这行在回调之后执行')
</script>
```

### 2.4 对比总结
```
sync  → 值变化时立即执行（同步）
pre   → 在 DOM 更新之前执行（异步批量）
post  → 在 DOM 更新之后执行（异步批量）
```

## 三、注意事项与常见陷阱
- `sync` 性能开销最大，一般不推荐使用
- 需要在回调中访问更新后的 DOM 时，使用 `flush: 'post'`
- `watchEffect` 的 flush 默认是 `pre`，可用 `watchEffect(cb, { flush: 'post' })` 修改
- 大多数场景使用默认的 `pre` 即可

## 四、flush 的典型使用场景

### 4.1 post：获取更新后的 DOM
```vue
<template>
  <ul ref="listRef">
    <li v-for="item in items" :key="item.id">{{ item.name }}</li>
  </ul>
</template>
<script setup>
import { ref, watch } from 'vue'

const items = ref([])
const listRef = ref(null)

watch(items, () => {
  // flush: 'post' 确保 DOM 已更新
  if (listRef.value) {
    listRef.value.scrollTop = listRef.value.scrollHeight
    console.log(`列表现在有 ${listRef.value.children.length} 个子元素`)
  }
}, { flush: 'post' })
</script>
```

### 4.2 sync：需要立即响应
```vue
<script setup>
import { ref, watch } from 'vue'

const inputValue = ref('')
const formattedValue = ref('')

// sync: 每次值变化立即格式化（性能敏感场景慎用）
watch(inputValue, (val) => {
  formattedValue.value = val.toUpperCase()
}, { flush: 'sync' })
</script>
```

### 4.3 watchEffect 的 flush
```vue
<template>
  <div ref="chartEl" style="width: 600px; height: 400px"></div>
</template>
<script setup>
import { ref, watchEffect } from 'vue'
import * as echarts from 'echarts'

const chartData = ref([])
const chartEl = ref(null)

// DOM 更新后初始化图表
watchEffect(() => {
  if (chartEl.value && chartData.value.length) {
    const chart = echarts.init(chartEl.value)
    chart.setOption({ series: [{ data: chartData.value }] })
  }
}, { flush: 'post' })
</script>
```

## 五、性能对比

```
sync → 每次变化立即执行，可能触发多次不必要的更新
pre  → 批量更新，在 DOM 更新前执行（默认，性能最佳）
post → 批量更新，在 DOM 更新后执行（需要访问新 DOM 时使用）
```

在高频更新场景（如拖拽、输入），`sync` 可能导致严重的性能问题。应优先使用 `pre` 或 `post`。
