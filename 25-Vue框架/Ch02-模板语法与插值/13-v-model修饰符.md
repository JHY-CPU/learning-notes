# v-model 修饰符

## 一、概念说明

Vue 为 `v-model` 提供了三个内置修饰符：`.trim`（自动去除首尾空格）、`.number`（自动转为数字）、`.lazy`（在 change 事件而非 input 事件时同步）。这些修饰符可以简化表单数据处理。

```vue
<script setup>
import { ref } from 'vue'

const text = ref('')
const age = ref(0)
const lazyText = ref('')
</script>

<template>
  <!-- .trim: 去除首尾空格 -->
  <input v-model.trim="text" />
  <p>输入: "{{ text }}"</p>

  <!-- .number: 自动转为数字 -->
  <input v-model.number="age" type="number" />
  <p>类型: {{ typeof age }}</p>

  <!-- .lazy: 失焦/回车时更新 -->
  <input v-model.lazy="lazyText" />
  <p>内容: {{ lazyText }}</p>
</template>
```

## 二、具体用法

### 2.1 .trim 修饰符

```vue
<script setup>
import { ref } from 'vue'
const username = ref('')
const email = ref('')
</script>

<template>
  <!-- 用户输入 "  hello  " → username 为 "hello" -->
  <input v-model.trim="username" placeholder="用户名" />
  <input v-model.trim="email" placeholder="邮箱" />
</template>
```

### 2.2 .number 修饰符

```vue
<script setup>
import { ref } from 'vue'
const price = ref(0)
</script>

<template>
  <!-- type="number" 返回的始终是字符串 -->
  <!-- .number 自动用 parseFloat 转换 -->
  <input v-model.number="price" type="number" />
  <p>{{ typeof price }} - {{ price }}</p>
  <!-- 没有 .number: "string" - "42" -->
  <!-- 有 .number: "number" - 42 -->
</template>
```

### 2.3 .lazy 修饰符

```vue
<script setup>
import { ref } from 'vue'
const search = ref('')
</script>

<template>
  <!-- 默认: 每次 input 事件都更新 -->
  <input v-model="search" />

  <!-- .lazy: 仅在 change 事件（失焦或回车）时更新 -->
  <input v-model.lazy="search" />
  <p>搜索: {{ search }}</p>
</template>
```

### 2.4 自定义修饰符

```vue
<!-- 父组件 -->
<template>
  <MyInput v-model.capitalize="text" />
</template>

<!-- 子组件 -->
<script setup>
const props = defineProps({
  modelValue: String,
  modelModifiers: { default: () => ({}) }
})
const emit = defineEmits(['update:modelValue'])

function handleInput(e) {
  let value = e.target.value
  if (props.modelModifiers.capitalize) {
    value = value.charAt(0).toUpperCase() + value.slice(1)
  }
  emit('update:modelValue', value)
}
</script>
```

## 三、注意事项与常见陷阱

- `.number` 无法转换非数字开头的字符串（如 "123abc" → 123）
- `.lazy` 在失去焦点或按回车时才更新数据
- 修饰符可以组合使用：`v-model.trim.number`
- 自定义组件中修饰符通过 `modelModifiers` prop 获取
- `.number` 对于非数字输入会返回原始字符串
