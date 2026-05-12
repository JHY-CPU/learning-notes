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

## 四、复杂Schema示例

```js
import * as yup from 'yup'

const registrationSchema = yup.object({
  // 基础信息
  username: yup.string()
    .required('用户名必填')
    .min(3, '至少3个字符')
    .max(20, '最多20个字符')
    .matches(/^[a-zA-Z0-9_]+$/, '只能包含字母、数字和下划线'),

  email: yup.string()
    .required('邮箱必填')
    .email('邮箱格式不正确'),

  password: yup.string()
    .required('密码必填')
    .min(8, '至少8位')
    .matches(/[A-Z]/, '需包含大写字母')
    .matches(/[0-9]/, '需包含数字'),

  confirmPassword: yup.string()
    .required('确认密码必填')
    .oneOf([yup.ref('password')], '两次密码不一致'),

  // 可选字段
  phone: yup.string()
    .nullable()
    .matches(/^1[3-9]\d{9}$/, '手机号格式不正确'),

  // 条件验证
  company: yup.string()
    .when('role', {
      is: 'enterprise',
      then: (schema) => schema.required('企业用户必须填写公司名'),
      otherwise: (schema) => schema.nullable()
    }),

  // 数组验证
  hobbies: yup.array()
    .of(yup.string())
    .min(1, '至少选择一个爱好')
    .max(5, '最多选择5个爱好'),

  // 对象验证
  address: yup.object({
    province: yup.string().required('省份必填'),
    city: yup.string().required('城市必填'),
    detail: yup.string().required('详细地址必填')
  })
})
```

## 五、Schema复用与组合

```js
// 基础Schema
const baseUserSchema = yup.object({
  name: yup.string().required().min(2),
  email: yup.string().required().email()
})

// 创建Schema（额外字段）
const createUserSchema = baseUserSchema.shape({
  password: yup.string().required().min(8),
  confirmPassword: yup.string()
    .required()
    .oneOf([yup.ref('password')], '密码不一致')
})

// 编辑Schema（密码可选）
const editUserSchema = baseUserSchema.shape({
  password: yup.string().min(8).nullable(),
  confirmPassword: yup.string()
    .oneOf([yup.ref('password')], '密码不一致')
    .nullable()
})
```

## 三、注意事项与常见陷阱

1. Yup的`.ref()`引用其他字段值（如确认密码）
2. Schema是纯数据，可复用和测试
3. 错误消息可通过`.message('自定义消息')`自定义
4. Yup支持自定义验证规则（`.test()`）
5. Schema可从后端获取，实现动态验证规则
6. `.when()`实现条件验证（根据其他字段的值）
7. `.shape()`可以扩展已有Schema，实现复用
