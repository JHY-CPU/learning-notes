# watchEffect

## 一、概念说明

`watchEffect()` 自动追踪回调函数中使用的所有响应式依赖，依赖变化时自动重新执行。与 `watch` 不同，不需要指定侦听源，适合依赖明确但来源分散的场景。

```vue
<script setup>
import { ref, watchEffect } from 'vue'

const count = ref(0)
const multiplier = ref(2)

// 自动追踪 count 和 multiplier
watchEffect(() => {
  console.log(`${count.value} × ${multiplier.value} = ${count.value * multiplier.value}`)
})
// 立即执行一次，之后依赖变化时自动执行
</script>

<template>
  <button @click="count++">count +1</button>
  <button @click="multiplier++">multiplier +1</button>
</template>
```

## 二、具体用法

### 2.1 自动依赖追踪

```vue
<script setup>
import { ref, reactive, watchEffect } from 'vue'

const userId = ref(1)
const userData = ref(null)

// 自动追踪 userId，不追踪 userData（只读取）
watchEffect(async () => {
  const response = await fetch(`/api/users/${userId.value}`)
  userData.value = await response.json()
})
</script>
```

### 2.2 副作用清理

```vue
<script setup>
import { ref, watchEffect } from 'vue'

const query = ref('')

watchEffect((onCleanup) => {
  const controller = new AbortController()

  // 清理函数：在下次执行前或停止时调用
  onCleanup(() => {
    controller.abort() // 取消上一次请求
  })

  fetch(`/api/search?q=${query.value}`, {
    signal: controller.signal
  })
})
</script>
```

### 2.3 停止 watchEffect

```vue
<script setup>
import { ref, watchEffect } from 'vue'

const count = ref(0)
const stop = watchEffect(() => {
  console.log(count.value)
})

// 手动停止
stop()

// 组件卸载时自动停止
</script>
```

### 2.4 flush 选项

```vue
<script setup>
import { ref, watchEffect } from 'vue'

const el = ref(null)

// flush: 'post' 确保在 DOM 更新后执行
watchEffect(() => {
  if (el.value) {
    console.log('DOM 元素高度:', el.value.offsetHeight)
  }
}, { flush: 'post' })
</script>
```

## 三、注意事项与常见陷阱

- watchEffect 在创建时立即执行（不像 watch 默认懒执行）
- 只追踪回调中实际使用的响应式依赖
- watchEffect 不访问旧值（只有当前值）
- 使用 `onCleanup` 清理异步副作用
- `flush: 'sync'` 没有批处理，可能影响性能
