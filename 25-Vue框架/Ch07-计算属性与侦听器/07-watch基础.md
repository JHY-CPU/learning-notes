# watch 基础

## 一、概念说明
`watch` 用于监听响应式数据的变化，在变化时执行**回调函数**。与 computed 不同，watch 用于执行副作用（如 API 请求、日志记录），而非返回计算值。

## 二、具体用法

### 2.1 侦听 ref
```vue
<script setup>
import { ref, watch } from 'vue'

const question = ref('')

watch(question, (newVal, oldVal) => {
  console.log(`问题从 "${oldVal}" 变为 "${newVal}"`)
})
</script>

<template>
  <input v-model="question" placeholder="输入问题" />
</template>
```

### 2.2 回调参数详解
```vue
<script setup>
import { ref, watch } from 'vue'

const count = ref(0)

watch(count, (newValue, oldValue) => {
  console.log(`新值: ${newValue}, 旧值: ${oldValue}`)
})
</script>
```

### 2.3 异步回调
```vue
<script setup>
import { ref, watch } from 'vue'

const searchQuery = ref('')

watch(searchQuery, async (query) => {
  if (!query) return
  const results = await fetch(`/api/search?q=${query}`)
    .then(r => r.json())
  console.log('搜索结果:', results)
})
</script>
```

### 2.4 立即执行 + 深度侦听
```vue
<script setup>
import { ref, watch } from 'vue'

const user = ref({ name: '张三', address: { city: '北京' } })

watch(user, (newVal) => {
  console.log('用户信息变化:', newVal)
}, { immediate: true, deep: true })
</script>
```

## 三、注意事项与常见陷阱
- 默认情况下 watch 是**懒执行**的，首次不会触发
- 使用 `{ immediate: true }` 让 watch 立即执行一次
- 侦听 ref 时回调接收 `(newVal, oldVal)`
- 侦听 reactive 对象时，oldVal 和 newVal 是同一个引用（需要 deep: true）
