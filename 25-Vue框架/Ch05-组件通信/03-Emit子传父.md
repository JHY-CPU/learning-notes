# Emit 子传父

## 一、概念说明
子组件通过**触发事件**（emit）将数据发送给父组件。父组件通过 `v-on` 监听这些事件并做出响应。这是一种反向的数据流，维持了 Vue 的单向数据流原则。

## 二、具体用法

### 2.1 基本用法
```vue
<!-- 子组件 Counter.vue -->
<template>
  <button @click="increment">+1</button>
</template>
<script setup>
const emit = defineEmits(['change'])

function increment() {
  emit('change', 1)  // 触发 change 事件，传递参数 1
}
</script>
```

```vue
<!-- 父组件 -->
<template>
  <p>当前值: {{ count }}</p>
  <Counter @change="handleCountChange" />
</template>
<script setup>
import { ref } from 'vue'
import Counter from './Counter.vue'
const count = ref(0)
const handleCountChange = (delta) => {
  count.value += delta
}
</script>
```

### 2.2 传递多个参数
```vue
<script setup>
const emit = defineEmits(['submit'])

function handleSubmit() {
  emit('submit', { username: '张三', age: 25 })
}
</script>

<!-- 父组件接收 -->
<!-- <Form @submit="(data) => console.log(data)" /> -->
```

### 2.3 TypeScript 类型化 emit
```vue
<script setup>
const emit = defineEmits<{
  change: [value: number]
  submit: [data: { username: string; age: number }]
}>()
</script>
```

## 三、注意事项与常见陷阱
- `defineEmits` 不需要导入，是编译器宏
- 事件名推荐使用 kebab-case（短横线命名）
- 不要在 emit 的回调中直接修改子组件自身状态
- 父组件可以用 `$event` 访问第一个事件参数
