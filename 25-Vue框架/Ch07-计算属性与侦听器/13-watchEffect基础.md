# watchEffect 基础

## 一、概念说明
`watchEffect` 会**立即执行**传入的函数，并**自动追踪**函数内使用的所有响应式依赖。当任何依赖变化时，函数会重新运行。与 `watch` 相比，不需要手动指定依赖。

## 二、具体用法

### 2.1 基本用法
```vue
<script setup>
import { ref, watchEffect } from 'vue'

const width = ref(100)
const height = ref(200)

// 自动追踪 width 和 height
watchEffect(() => {
  console.log(`尺寸: ${width.value}x${height.value}`)
})
// 立即输出: 尺寸: 100x200
// 修改 width 后自动重新执行
</script>
```

### 2.2 DOM 更新后执行
```vue
<template>
  <div ref="el">{{ text }}</div>
</template>
<script setup>
import { ref, watchEffect } from 'vue'

const text = ref('Hello')
const el = ref(null)

watchEffect(() => {
  // 自动追踪 text 变化
  if (el.value) {
    console.log('DOM 内容:', el.value.textContent)
  }
}, { flush: 'post' })
</script>
```

### 2.3 异步副作用
```vue
<script setup>
import { ref, watchEffect } from 'vue'

const userId = ref(1)
const user = ref(null)

watchEffect(async () => {
  user.value = await fetch(`/api/users/${userId.value}`)
    .then(r => r.json())
})
</script>
```

### 2.4 与 watch 的选择
```js
// 需要 oldVal → watch
watch(count, (newVal, oldVal) => { /* ... */ })

// 自动追踪多个依赖 → watchEffect
watchEffect(() => {
  console.log(count.value, name.value)  // 自动追踪两个依赖
})
```

## 三、注意事项与常见陷阱
- `watchEffect` 立即执行，`watch` 默认懒执行
- `watchEffect` 无法获取 `oldValue`
- 不要在 watchEffect 中做异步操作后依赖异步结果的响应式追踪
- 追踪是基于函数执行期间访问的响应式属性
