# VeeValidate与Yup

## 一、概念说明

Yup是一个schema验证库，与VeeValidate配合实现声明式表单验证。Schema定义验证规则，自动验证表单字段。

```bash
npm install yup
```

```vue
<template>
  <Form :validation-schema="schema" @submit="onSubmit">
    <Field name="name" />
    <ErrorMessage name="name" />

    <Field name="email" type="email" />
    <ErrorMessage name="email" />

    <Field name="password" type="password" />
    <ErrorMessage name="password" />

    <button>注册</button>
  </Form>
</template>

<script setup>
import { Form, Field, ErrorMessage } from 'vee-validate'
import * as yup from 'yup'

const schema = yup.object({
  name: yup.string().required('姓名必填').min(2, '至少2个字符'),
  email: yup.string().required('邮箱必填').email('邮箱格式不正确'),
  password: yup.string().required('密码必填').min(6, '至少6位')
})

const onSubmit = (values) => {
  console.log(values)
}
</script>
```

## 二、具体用法

### 常用Yup规则

```js
const schema = yup.object({
  username: yup.string().required().min(3).max(20),
  email: yup.string().required().email(),
  age: yup.number().required().min(0).max(150).integer(),
  website: yup.string().url().nullable(),
  password: yup.string().required().min(6),
  confirmPassword: yup.string()
    .required()
    .oneOf([yup.ref('password')], '密码不一致'),
  agree: yup.boolean().oneOf([true], '必须同意条款')
})
```

## 三、注意事项与常见陷阱

1. Yup的`.ref()`引用其他字段值（如确认密码）
2. Schema是纯数据，可复用和测试
3. 错误消息可通过`.message('自定义消息')`自定义
4. Yup支持自定义验证规则（`.test()`）
5. Schema可从后端获取，实现动态验证规则
