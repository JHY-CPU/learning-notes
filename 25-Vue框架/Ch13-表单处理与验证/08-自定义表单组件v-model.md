# 自定义表单组件v-model

## 一、概念说明

`defineModel`是Vue 3.4+的宏，简化自定义组件的`v-model`实现。它自动处理props和emit。

```vue
<!-- CustomInput.vue -->
<template>
  <input :value="modelValue" @input="modelValue = $event.target.value" />
</template>

<script setup>
// Vue 3.4+ 简洁写法
const modelValue = defineModel()
</script>

<!-- 使用 -->
<template>
  <CustomInput v-model="text" />
</template>
```

## 二、具体用法

### 带参数的defineModel

```vue
<!-- 自定义组件 -->
<script setup>
const modelValue = defineModel({ type: String, default: '' })
const count = defineModel('count', { type: Number, default: 0 })
</script>

<!-- 使用 -->
<template>
  <MyComponent v-model="name" v-model:count="num" />
</template>
```

### 自定义修饰符

```vue
<script setup>
const [modelValue, modifiers] = defineModel({
  set(value) {
    // 处理.trim修饰符
    if (modifiers.trim) return value.trim()
    return value
  }
})
</script>

<!-- 使用 -->
<template>
  <CustomInput v-model.trim="text" />
</template>
```

### 手动实现（兼容3.4之前）

```vue
<script setup>
const props = defineProps({ modelValue: String })
const emit = defineEmits(['update:modelValue'])

const onInput = (e) => {
  emit('update:modelValue', e.target.value)
}
</script>

<template>
  <input :value="props.modelValue" @input="onInput" />
</template>
```

## 三、注意事项与常见陷阱

1. `defineModel`是Vue 3.4+特性，旧版本需手动实现
2. `defineModel`返回的是ref，在模板中自动解包
3. 多个v-model用不同名称（`v-model:xxx`）
4. 自定义组件的v-model默认prop是`modelValue`
5. 修饰符通过defineModel的第二个返回值获取
