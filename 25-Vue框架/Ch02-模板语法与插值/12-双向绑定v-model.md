# 双向绑定 v-model

## 一、概念说明

`v-model` 是 Vue 提供的语法糖，用于在表单元素上创建**双向数据绑定**。它本质上是 `:value` + `@input`（或 `:modelValue` + `@update:modelValue`）的简写。

```vue
<script setup>
import { ref } from 'vue'

const message = ref('')
const selected = ref('')
const checked = ref(false)
</script>

<template>
  <!-- 文本输入 -->
  <input v-model="message" placeholder="输入内容" />
  <p>输入的内容: {{ message }}</p>

  <!-- 下拉选择 -->
  <select v-model="selected">
    <option value="a">选项 A</option>
    <option value="b">选项 B</option>
  </select>
</template>
```

## 二、具体用法

### 2.1 各种表单元素

```vue
<script setup>
import { ref } from 'vue'
const text = ref('')
const multiline = ref('')
const checked = ref(false)
const picked = ref('one')
const selected = ref([])
</script>

<template>
  <!-- 文本 -->
  <input v-model="text" />

  <!-- 多行文本 -->
  <textarea v-model="multiline"></textarea>

  <!-- 复选框（单个 - 布尔值） -->
  <input type="checkbox" v-model="checked" />

  <!-- 复选框（多个 - 数组） -->
  <input type="checkbox" value="Vue" v-model="selected" />
  <input type="checkbox" value="React" v-model="selected" />

  <!-- 单选按钮 -->
  <input type="radio" value="one" v-model="picked" />
  <input type="radio" value="two" v-model="picked" />

  <!-- 选择框 -->
  <select v-model="selected">
    <option value="a">A</option>
    <option value="b">B</option>
  </select>
</template>
```

### 2.2 自定义组件的 v-model

```vue
<!-- 子组件: CustomInput.vue -->
<script setup>
const props = defineProps(['modelValue'])
const emit = defineEmits(['update:modelValue'])
</script>

<template>
  <input
    :value="modelValue"
    @input="emit('update:modelValue', $event.target.value)"
  />
</template>

<!-- 父组件使用 -->
<template>
  <CustomInput v-model="message" />
</template>
```

## 三、注意事项与常见陷阱

- `v-model` 默认绑定 `modelValue` prop 和 `update:modelValue` 事件
- 复选框绑定数组时，`value` 决定数组中的值
- `v-model` 在 `<input type="file">` 上不生效
- `textarea` 不能使用 `{{ }}` 插值作为内容
- 自定义组件实现 `v-model` 时需要正确声明 props 和 emits
