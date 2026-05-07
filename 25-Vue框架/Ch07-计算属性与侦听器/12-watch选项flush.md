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
