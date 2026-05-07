# 多 v-model 绑定

## 一、概念说明
Vue 3 支持在同一个组件上使用**多个 v-model**，通过指定不同的名称（如 `v-model:title`），可以在一个组件上实现多组双向绑定。

## 二、具体用法

### 2.1 基本语法
```vue
<!-- 父组件 -->
<template>
  <UserForm
    v-model:firstName="first"
    v-model:lastName="last"
  />
  <p>全名: {{ first }} {{ last }}</p>
</template>
<script setup>
import { ref } from 'vue'
import UserForm from './UserForm.vue'
const first = ref('张')
const last = ref('三')
</script>
```

### 2.2 子组件实现
```vue
<!-- UserForm.vue -->
<template>
  <input
    :value="firstName"
    @input="$emit('update:firstName', $event.target.value)"
    placeholder="姓"
  />
  <input
    :value="lastName"
    @input="$emit('update:lastName', $event.target.value)"
    placeholder="名"
  />
</template>
<script setup>
defineProps<{
  firstName: string
  lastName: string
}>()
defineEmits(['update:firstName', 'update:lastName'])
</script>
```

### 2.3 使用 defineModel 简化（Vue 3.3+）
```vue
<script setup>
const firstName = defineModel<string>('firstName')
const lastName = defineModel<string>('lastName')
</script>

<template>
  <input v-model="firstName" placeholder="姓" />
  <input v-model="lastName" placeholder="名" />
</template>
```

### 2.4 带默认值的 defineModel
```vue
<script setup>
const title = defineModel('title', { default: '默认标题' })
const content = defineModel('content', { default: '' })
</script>
```

## 三、注意事项与常见陷阱
- 多 v-model 的命名约定与 props 一致，推荐 camelCase
- `defineModel` 是 Vue 3.3+ 的实验性特性，需要确认项目版本
- 每个 v-model 都需要对应的 `update:xxx` 事件
- 避免在一个组件上绑定过多 v-model，超过 3 个考虑用对象形式
