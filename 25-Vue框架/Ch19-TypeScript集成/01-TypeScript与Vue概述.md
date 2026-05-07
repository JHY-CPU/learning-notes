# TypeScript与Vue概述

## 一、概念说明

TypeScript 是 JavaScript 的超集，为 Vue 3 提供静态类型检查。Vue 3 本身使用 TypeScript 重写，对 TS 有一等支持。使用 TS 可以在编码阶段发现错误、获得更好的 IDE 自动补全、提升代码可维护性。Vue 3 + Vite + TypeScript 是当前推荐的技术栈。

## 二、具体用法

### 为什么用 TypeScript

```ts
// 无 TypeScript：运行时才发现错误
function formatUser(user) {
  return `${user.firstName} ${user.lastName}`
}
formatUser({ name: '张三' })  // 运行时返回 'undefined undefined'

// 有 TypeScript：编码时就能发现错误
interface User {
  firstName: string
  lastName: string
  age: number
}

function formatUser(user: User): string {
  return `${user.firstName} ${user.lastName}`
}
formatUser({ name: '张三' })
// 编译错误：类型 "{ name: string }" 中缺少属性 "firstName, lastName"
```

### Vue 3 中的 TS 优势

```vue
<script setup lang="ts">
// Props 自动类型推断
interface Props {
  title: string
  count: number
  items: string[]
  onSelect?: (id: number) => void
}

const props = defineProps<Props>()
// props.title 类型为 string
// props.count 类型为 number
// props.items 类型为 string[]

// 事件类型安全
const emit = defineEmits<{
  submit: [data: { name: string; email: string }]
  cancel: []
}>()

function handleSubmit() {
  emit('submit', { name: '张三', email: 'test@example.com' })
  // emit('submit', { wrong: 'data' })
  // 类型错误：不能将 { wrong: string } 赋给 { name: string; email: string }
}
</script>

<template>
  <div>
    <h1>{{ title }}</h1>
    <p>计数: {{ count }}</p>
    <ul>
      <li v-for="item in items" :key="item">{{ item }}</li>
    </ul>
    <button @click="handleSubmit">提交</button>
  </div>
</template>
```

### Vue 3 类型支持层级

```text
Vue 3 类型支持金字塔：

1. Props 类型       → defineProps<T>()
2. Emit 类型        → defineEmits<T>()
3. Ref/Reactive 类型 → Ref<T>, Reactive<T>
4. Computed 类型     → ComputedRef<T>
5. 组件类型          → DefineComponent
6. 模板引用类型      → Ref<InstanceType<typeof Comp>>
7. 组合式函数类型    → 返回值类型定义
8. Store 类型        → DefineStore 泛型
```

## 三、注意事项与常见陷阱

1. **Vue 3.3+ 才支持 defineProps 泛型语法**：旧版本需要使用运行时声明
2. **`<script setup lang="ts">` 是必须的**：不加 lang="ts" 则 TypeScript 语法不生效
3. **不需要手动定义 props 类型**：defineProps 的泛型会自动推断类型
4. **TS 不影响运行时性能**：类型检查只在编译时进行，运行时被擦除
5. **从 JavaScript 迁移可以渐进式**：允许 .js 和 .ts 文件共存
