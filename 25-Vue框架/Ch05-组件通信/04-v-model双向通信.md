# v-model 双向通信

## 一、概念说明
`v-model` 是 Vue 提供的语法糖，用于在父子组件之间建立**双向数据绑定**。它本质上是 `:modelValue` prop 和 `@update:modelValue` emit 的组合。

## 二、具体用法

### 2.1 基本原理
```vue
<!-- 以下两种写法等价 -->
<MyInput v-model="username" />

<MyInput
  :modelValue="username"
  @update:modelValue="username = $event"
/>
```

### 2.2 子组件实现 v-model
```vue
<!-- MyInput.vue -->
<template>
  <input
    :value="modelValue"
    @input="$emit('update:modelValue', $event.target.value)"
  />
</template>
<script setup>
defineProps<{ modelValue: string }>()
defineEmits(['update:modelValue'])
</script>
```

### 2.3 使用 computed 简化
```vue
<script setup>
import { computed } from 'vue'
const props = defineProps<{ modelValue: string }>()
const emit = defineEmits(['update:modelValue'])

const value = computed({
  get: () => props.modelValue,
  set: (val) => emit('update:modelValue', val)
})
</script>

<template>
  <input v-model="value" />
</template>
```

### 2.4 修饰符支持
```vue
<!-- 父组件 -->
<MyInput v-model.trim="username" />
```

```vue
<!-- 子组件接收 modifiers -->
<script setup>
const props = defineProps<{
  modelValue: string
  modelModifiers?: { trim?: boolean }
}>()
const emit = defineEmits(['update:modelValue'])

function onInput(e) {
  let val = e.target.value
  if (props.modelModifiers?.trim) val = val.trim()
  emit('update:modelValue', val)
}
</script>
```

### 2.5 defineModel（Vue 3.3+）
```vue
<script setup>
const model = defineModel()
</script>

<template>
  <input v-model="model" />
</template>
```

## 三、常见用例

| 场景 | 实现方式 |
|------|---------|
| 自定义输入框 | v-model + input 事件 |
| 开关组件 | v-model + boolean |
| 下拉选择器 | v-model + change 事件 |
| 带格式化的输入 | v-model + 修饰符 |

## 四、注意事项与常见陷阱

- Vue 3.3+ 支持 `defineModel()` 宏简化实现
- v-model 默认绑定的 prop 名是 `modelValue`
- 修饰符需要子组件自行处理，不会自动生效
- 简单表单控件可直接用 v-model，复杂逻辑建议用 computed 包装
- v-model 不会自动处理复杂对象的深层变更
- 避免在 v-model 的 setter 中执行异步操作
