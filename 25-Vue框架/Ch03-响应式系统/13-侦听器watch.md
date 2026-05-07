# 侦听器 watch

## 一、概念说明

`watch()` 侦听一个或多个响应式数据源，在数据变化时执行回调函数。与 computed 不同，watch 适合执行副作用（如异步操作、DOM 操作）。

```vue
<script setup>
import { ref, watch } from 'vue'

const question = ref('')
const answer = ref('等待提问...')

// 侦听 question 的变化
watch(question, async (newQuestion) => {
  if (newQuestion.includes('?')) {
    answer.value = '思考中...'
    // 模拟 API 调用
    setTimeout(() => {
      answer.value = '答案是 42'
    }, 500)
  }
})
</script>

<template>
  <input v-model="question" placeholder="输入问题（以?结尾）" />
  <p>{{ answer }}</p>
</template>
```

## 二、具体用法

### 2.1 侦听 ref

```vue
<script setup>
import { ref, watch } from 'vue'

const count = ref(0)

watch(count, (newVal, oldVal) => {
  console.log(`count: ${oldVal} → ${newVal}`)
})
</script>
```

### 2.2 侦听 reactive

```vue
<script setup>
import { reactive, watch } from 'vue'

const state = reactive({ count: 0, name: 'Vue' })

// 侦听整个 reactive 对象
watch(state, (newVal) => {
  console.log('state 变化了:', newVal)
})

// 侦听 reactive 的某个属性（用 getter）
watch(
  () => state.count,
  (newVal, oldVal) => {
    console.log(`count: ${oldVal} → ${newVal}`)
  }
)
</script>
```

### 2.3 侦听多个源

```vue
<script setup>
import { ref, watch } from 'vue'

const a = ref(1)
const b = ref(2)

// 侦听多个 ref
watch([a, b], ([newA, newB], [oldA, oldB]) => {
  console.log(`a: ${oldA}→${newA}, b: ${oldB}→${newB}`)
})
</script>
```

### 2.4 停止侦听

```vue
<script setup>
import { ref, watch } from 'vue'

const count = ref(0)
const stop = watch(count, () => {
  console.log('count changed')
})

// 停止侦听
stop()
</script>
```

## 三、注意事项与常见陷阱

- watch 默认是懒执行的，数据变化后才触发
- 侦听 reactive 对象时，默认是浅层的（不追踪嵌套变化）
- 用 `() => state.count` 而不是 `state.count` 作为侦听源
- watch 回调中可以访问 `onCleanup` 清理过期的异步操作
- 组件卸载时自动停止侦听
