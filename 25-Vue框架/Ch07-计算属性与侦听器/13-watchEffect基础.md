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

## 四、watchEffect 的高级用法

### 4.1 同步 DOM 测量
```vue
<template>
  <div ref="el" :style="{ width: width + 'px' }">
    {{ content }}
  </div>
  <p>实际宽度: {{ actualWidth }}px</p>
</template>
<script setup>
import { ref, watchEffect } from 'vue'

const width = ref(200)
const content = ref('测试内容')
const el = ref(null)
const actualWidth = ref(0)

watchEffect(() => {
  if (el.value) {
    actualWidth.value = el.value.offsetWidth
  }
}, { flush: 'post' })
</script>
```

### 4.2 自动事件监听
```vue
<script setup>
import { ref, watchEffect } from 'vue'

const mousePos = ref({ x: 0, y: 0 })
const isTracking = ref(true)

watchEffect((onInvalidate) => {
  if (!isTracking.value) return

  const handler = (e) => {
    mousePos.value = { x: e.clientX, y: e.clientY }
  }
  window.addEventListener('mousemove', handler)

  onInvalidate(() => {
    window.removeEventListener('mousemove', handler)
  })
})
</script>
```

### 4.3 同步数据验证
```vue
<script setup>
import { ref, watchEffect } from 'vue'

const email = ref('')
const errors = ref([])

watchEffect(() => {
  const newErrors = []
  if (email.value && !email.value.includes('@')) {
    newErrors.push('邮箱格式不正确')
  }
  if (email.value && email.value.length > 50) {
    newErrors.push('邮箱长度不能超过 50 个字符')
  }
  errors.value = newErrors
})
</script>
```

## 五、watchEffect vs watch 详细对比

| 特性 | watchEffect | watch |
|------|-------------|-------|
| 依赖声明 | 自动追踪 | 手动指定 |
| 立即执行 | 是 | 需 `immediate` |
| oldValue | 不可用 | 可用 |
| 清理函数 | `onInvalidate` | `onCleanup`（3.4+）|
| flush | 默认 pre | 默认 pre |
| 适用场景 | DOM 副作用、自动追踪 | 特定数据监听、需要旧值 |
