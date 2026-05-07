# 组件事件 emit

## 一、概念说明

`emit` 是子组件向父组件通信的方式。子组件通过 `defineEmits` 声明可触发的事件，使用 `emit()` 触发事件并传递数据。父组件通过 `@事件名` 监听事件。

```vue
<!-- 子组件 -->
<script setup>
const emit = defineEmits(['submit', 'cancel'])

function handleSubmit() {
  emit('submit', { name: '张三', age: 25 })
}

function handleCancel() {
  emit('cancel')
}
</script>

<template>
  <button @click="handleSubmit">提交</button>
  <button @click="handleCancel">取消</button>
</template>

<!-- 父组件 -->
<!-- <MyForm @submit="onSubmit" @cancel="onCancel" /> -->
```

## 二、具体用法

### 2.1 运行时声明

```vue
<script setup>
const emit = defineEmits(['change', 'submit', 'delete'])

function handleChange(value) {
  emit('change', value)
}
</script>
```

### 2.2 TypeScript 类型声明

```vue
<script setup lang="ts">
const emit = defineEmits<{
  change: [value: string]
  submit: [data: { name: string; email: string }]
  delete: [id: number]
}>()

// 触发事件，参数类型被检查
emit('change', 'hello')
emit('submit', { name: '张三', email: 'a@b.com' })
emit('delete', 42)
</script>
```

### 2.3 事件验证

```vue
<script setup>
const emit = defineEmits({
  // 无验证
  submit: null,

  // 带验证
  change: (value) => {
    if (typeof value !== 'string') {
      console.warn('change 事件需要字符串参数')
      return false
    }
    return true
  }
})
</script>
```

### 2.4 父组件监听

```vue
<!-- 父组件 -->
<script setup>
function handleSubmit(data) {
  console.log('收到提交:', data)
}
</script>

<template>
  <MyForm @submit="handleSubmit" />
</template>
```

## 三、注意事项与常见陷阱

- `defineEmits` 只能在 `<script setup>` 中使用
- 声明的事件名应该使用 kebab-case
- 事件验证只在开发模式下生效
- 不要通过 emit 传递复杂对象（保持接口简洁）
- 组件的 events 选项在 Vue 3 中已移除，使用 `defineEmits`
