# 模板 ref 获取子组件

## 一、概念说明
通过模板 `ref`，父组件可以直接**获取子组件实例**并调用其暴露的方法或访问数据。Vue 3 的 `<script setup>` 默认是封闭的，需要使用 `defineExpose` 显式暴露成员。

## 二、具体用法

### 2.1 子组件 defineExpose
```vue
<!-- ChildForm.vue -->
<template>
  <input ref="inputRef" v-model="value" />
</template>
<script setup>
import { ref } from 'vue'

const value = ref('')
const inputRef = ref(null)

function validate() {
  return value.value.length > 0
}

function focus() {
  inputRef.value?.focus()
}

// 暴露给父组件的方法和数据
defineExpose({ validate, focus, value })
</script>
```

### 2.2 父组件使用 ref
```vue
<template>
  <ChildForm ref="formRef" />
  <button @click="submitForm">提交</button>
</template>
<script setup>
import { ref } from 'vue'
import ChildForm from './ChildForm.vue'

const formRef = ref(null)

function submitForm() {
  if (formRef.value.validate()) {
    console.log('表单值:', formRef.value.value)
  } else {
    formRef.value.focus()
  }
}
</script>
```

### 2.3 模板 ref 数组（v-for 中）
```vue
<template>
  <input v-for="i in 3" :key="i" ref="inputRefs" />
</template>
<script setup>
import { ref, onMounted } from 'vue'
const inputRefs = ref([])

onMounted(() => {
  inputRefs.value.forEach(el => console.log(el))
})
</script>
```

## 三、注意事项与常见陷阱
- **必须使用 `defineExpose`**，否则父组件无法访问子组件内部成员
- 避免过度依赖模板 ref，它破坏了组件的封装性
- ref 在 `onMounted` 之后才可用，模板中使用需要加 `v-if` 保护
- 尽量只暴露方法而非内部状态
