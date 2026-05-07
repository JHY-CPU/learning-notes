# 组件 v-model

## 一、概念说明

`v-model` 在组件上使用时，本质上是 `:modelValue` + `@update:modelValue` 的语法糖。Vue 3 支持多个 `v-model` 绑定（`v-model:title`、`v-model:content`）。

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

<!-- 父组件 -->
<!-- <CustomInput v-model="message" /> -->
<!-- 等价于: <CustomInput :modelValue="message" @update:modelValue="message = $event" /> -->
```

## 二、具体用法

### 2.1 单个 v-model

```vue
<!-- 子组件: Toggle.vue -->
<script setup>
const props = defineProps({ modelValue: Boolean })
const emit = defineEmits(['update:modelValue'])
</script>

<template>
  <button @click="emit('update:modelValue', !modelValue)">
    {{ modelValue ? '开' : '关' }}
  </button>
</template>

<!-- 父组件 -->
<!-- <Toggle v-model="isOn" /> -->
```

### 2.2 多个 v-model

```vue
<!-- 子组件: UserForm.vue -->
<script setup>
defineProps({
  firstName: String,
  lastName: String
})
defineEmits(['update:firstName', 'update:lastName'])
</script>

<template>
  <input :value="firstName" @input="$emit('update:firstName', $event.target.value)" />
  <input :value="lastName" @input="$emit('update:lastName', $event.target.value)" />
</template>

<!-- 父组件 -->
<!-- <UserForm v-model:first-name="first" v-model:last-name="last" /> -->
```

### 2.3 带修饰符的 v-model

```vue
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

<!-- 父组件: <MyInput v-model.capitalize="text" /> -->
```

## 三、注意事项与常见陷阱

- `v-model` 默认绑定 `modelValue` prop
- 多个 v-model 使用 `v-model:propName` 语法
- 修饰符通过 `modelModifiers` prop 获取
- 原生 `v-model` 在组件上不会自动工作，需要手动实现
- 不要同时使用 v-model 和 modelValue prop 的单独绑定
