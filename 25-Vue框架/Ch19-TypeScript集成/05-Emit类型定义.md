# Emit类型定义

## 一、概念说明

`defineEmits` 的泛型语法允许用 TypeScript 定义组件发出事件的名称和参数类型。父组件监听事件时能获得完整的类型检查和自动补全。这确保了组件间通信的类型安全。

## 二、具体用法

### 基本 Emit 泛型

```vue
<script setup lang="ts">
// 方式1：调用签名语法
const emit = defineEmits<{
  change: [value: string]
  submit: [data: { name: string; email: string }]
  cancel: []
  delete: [id: number]
}>()

// emit('change', 123)     → 类型错误：number 不能赋给 string
// emit('submit', { name: '张三' }) → 类型错误：缺少 email
// emit('cancel')          → 正确
// emit('delete', 42)      → 正确

function handleChange(newVal: string) {
  emit('change', newVal)
  // 触发事件，父组件接收字符串参数
}

function handleSubmit() {
  emit('submit', { name: '李四', email: 'li@example.com' })
}

function handleCancel() {
  emit('cancel')
}
</script>

<template>
  <div>
    <input @input="handleChange($event.target.value)" />
    <button @click="handleSubmit">提交</button>
    <button @click="handleCancel">取消</button>
  </div>
</template>
```

### 表单组件完整示例

```vue
<script setup lang="ts">
interface FormData {
  username: string
  email: string
  age: number
}

const emit = defineEmits<{
  submit: [data: FormData]
  validate: [field: string, valid: boolean]
  'update:modelValue': [value: FormData]
}>()

const form = reactive<FormData>({
  username: '',
  email: '',
  age: 0
})

function handleFormSubmit() {
  emit('submit', { ...form })
  // emit('submit', { username: 'test' })
  // 编译错误：缺少 email 和 age
}

function validateField(field: keyof FormData) {
  const valid = form[field] !== '' && form[field] !== 0
  emit('validate', field, valid)
}
</script>

<template>
  <form @submit.prevent="handleFormSubmit">
    <input v-model="form.username" @blur="validateField('username')" />
    <input v-model="form.email" @blur="validateField('email')" />
    <input v-model.number="form.age" type="number" />
    <button type="submit">提交</button>
  </form>
</template>
```

### 组件使用时的类型检查

```vue
<!-- 父组件 -->
<script setup lang="ts">
function handleSubmit(data: { username: string; email: string; age: number }) {
  console.log('表单数据:', data)
  // 输出：表单数据: { username: "张三", email: "zhang@test.com", age: 25 }
}

function handleValidate(field: string, valid: boolean) {
  console.log(`${field} 验证: ${valid ? '通过' : '失败'}`)
}
</script>

<template>
  <UserForm
    @submit="handleSubmit"
    @validate="handleValidate"
  />
  <!-- @unknown-event="..." → 类型错误：事件不存在 -->
</template>
```

## 三、注意事项与常见陷阱

1. **调用签名语法是 Vue 3.3+ 特性**：旧版本使用 `defineEmits<{ (e: 'change', val: string): void }>()`
2. **数组语法定义参数类型**：`[value: string]` 中的名称仅供文档用途，类型检查依赖类型
3. **原生 DOM 事件不受 defineEmits 约束**：`@click` 等是原生事件，不需要声明
4. **`update:modelValue` 需要显式声明**：使用 v-model 的组件必须声明此事件
5. **Emit 类型不检查返回值**：事件处理函数的返回值类型不被约束
