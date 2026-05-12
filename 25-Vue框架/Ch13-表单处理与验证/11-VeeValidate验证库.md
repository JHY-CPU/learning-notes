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

## 四、完整注册表单示例

```vue
<template>
  <Form :validation-schema="schema" @submit="onSubmit" v-slot="{ errors, isSubmitting }">
    <div class="form-group">
      <label for="name">姓名</label>
      <Field id="name" name="name" as="input" class="form-control" />
      <ErrorMessage name="name" class="error" />
    </div>

    <div class="form-group">
      <label for="email">邮箱</label>
      <Field id="email" name="email" type="email" as="input" class="form-control" />
      <ErrorMessage name="email" class="error" />
    </div>

    <div class="form-group">
      <label for="password">密码</label>
      <Field id="password" name="password" type="password" as="input" class="form-control" />
      <ErrorMessage name="password" class="error" />
    </div>

    <button type="submit" :disabled="isSubmitting || Object.keys(errors).length > 0">
      {{ isSubmitting ? '注册中...' : '注册' }}
    </button>
  </Form>
</template>

<script setup>
import { Form, Field, ErrorMessage } from 'vee-validate'
import { required, email, min } from '@vee-validate/rules'
import { defineRule, configure } from 'vee-validate'

// 注册规则
defineRule('required', required)
defineRule('email', email)
defineRule('min', min)

// 全局配置
configure({
  generateMessage: (ctx) => {
    const messages = {
      required: `${ctx.field}不能为空`,
      email: '请输入有效的邮箱地址',
      min: `${ctx.field}至少需要${ctx.rule.params[0]}个字符`
    }
    return messages[ctx.rule.name] || `${ctx.field}验证失败`
  }
})

const schema = {
  name: 'required|min:2',
  email: 'required|email',
  password: 'required|min:8'
}

const onSubmit = (values) => {
  console.log('注册数据:', values)
}
</script>
```

## 五、表单级别的作用域插槽

```vue
<template>
  <Form v-slot="{ values, errors, setFieldValue, resetForm }">
    <Field name="score" as="input" type="number" />
    <button type="button" @click="setFieldValue('score', 100)">设为满分</button>
    <p>当前分数: {{ values.score }}</p>
    <button type="button" @click="resetForm()">重置</button>
  </Form>
</template>
```

## 三、注意事项与常见陷阱

1. VeeValidate 4专用于Vue 3
2. `Field`组件渲染为`<input>`，可指定`as`属性改变元素
3. `ErrorMessage`显示对应字段的错误消息
4. 规则字符串用`|`分隔，如`required|email|min:6`
5. 使用`defineRule`注册自定义规则
6. 全局配置`generateMessage`可统一错误消息格式
7. `resetForm()`可重置表单到初始状态
