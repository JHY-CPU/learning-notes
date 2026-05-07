# VeeValidate验证库

## 一、概念说明

VeeValidate是Vue的表单验证库，提供声明式验证规则、错误消息管理和表单状态。

```bash
npm install vee-validate @vee-validate/rules
```

```vue
<template>
  <Form @submit="onSubmit">
    <Field name="email" type="email" :rules="isRequired" />
    <ErrorMessage name="email" />
    <button>提交</button>
  </Form>
</template>

<script setup>
import { Form, Field, ErrorMessage } from 'vee-validate'

const isRequired = (value) => value ? true : '此字段必填'

const onSubmit = (values) => {
  console.log('提交:', values)
}
```

## 二、具体用法

### 使用内置规则

```vue
<script setup>
import { Form, Field, ErrorMessage } from 'vee-validate'
import { required, email, min } from '@vee-validate/rules'
import { defineRule } from 'vee-validate'

defineRule('required', required)
defineRule('email', email)
defineRule('min', min)
</script>

<template>
  <Form>
    <Field name="email" rules="required|email" />
    <ErrorMessage name="email" />

    <Field name="password" rules="required|min:6" />
    <ErrorMessage name="password" />
  </Form>
</template>
```

### 表单状态

```vue
<template>
  <Form v-slot="{ errors, isSubmitting, handleSubmit }">
    <Field name="name" rules="required" />
    <span>{{ errors.name }}</span>

    <button :disabled="isSubmitting || Object.keys(errors).length">
      {{ isSubmitting ? '提交中...' : '提交' }}
    </button>
  </Form>
</template>
```

## 三、注意事项与常见陷阱

1. VeeValidate 4专用于Vue 3
2. `Field`组件渲染为`<input>`，可指定`as`属性改变元素
3. `ErrorMessage`显示对应字段的错误消息
4. 规则字符串用`|`分隔，如`required|email|min:6`
5. 使用`defineRule`注册自定义规则
