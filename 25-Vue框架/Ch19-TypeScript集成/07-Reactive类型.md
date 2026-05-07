# Reactive类型

## 一、概念说明

`reactive()` 返回 `Reactive<T>` 类型，自动推断传入对象的类型。与 ref 不同，reactive 包装的对象直接访问属性（无需 `.value`），但有类型限制：只能代理对象类型（Object、Array、Map、Set），不能代理基本类型。

## 二、具体用法

### 基本类型推断

```ts
import { reactive } from 'vue'

// 自动推断对象类型
const state = reactive({
  count: 0,
  name: 'Vue',
  items: [1, 2, 3]
})
// state 类型: Reactive<{ count: number; name: string; items: number[] }>
// 直接访问：state.count, state.name, state.items

state.count++
// 类型安全：state.count 是 number，可以 ++
// state.count = 'hello' → 类型错误
```

### 接口定义

```ts
import { reactive } from 'vue'

interface FormState {
  username: string
  email: string
  password: string
  errors: Record<string, string>
  isSubmitting: boolean
}

const form = reactive<FormState>({
  username: '',
  email: '',
  password: '',
  errors: {},
  isSubmitting: false
})

// 重置表单
function resetForm() {
  form.username = ''
  form.email = ''
  form.password = ''
  form.errors = {}
  form.isSubmitting = false
}
```

### 与 toRefs 配合使用

```vue
<script setup lang="ts">
import { reactive, toRefs } from 'vue'

interface State {
  count: number
  message: string
}

const state = reactive<State>({
  count: 0,
  message: '你好'
})

// toRefs 保持解构后的响应性
const { count, message } = toRefs(state)
// count 类型: Ref<number>
// message 类型: Ref<string>

// 可以安全地传递给子组件
function increment() {
  count.value++
}
</script>

<template>
  <div>
    <p>计数: {{ count }}</p>
    <p>消息: {{ message }}</p>
    <button @click="increment">+1</button>
    <input v-model="message" />
  </div>
</template>
```

### Reactive 的局限

```ts
import { reactive } from 'vue'

// 错误：不能重新赋值整个对象（会失去响应性）
let state = reactive({ count: 0 })
// state = reactive({ count: 1 })  ← 错误做法

// 正确：修改属性
state.count = 1

// 错误：不能解构直接使用（会丢失响应性）
const { count } = reactive({ count: 0 })
// count 不再是响应式的

// 正确：使用 toRefs 解构
const state2 = reactive({ count: 0 })
const { count: count2 } = toRefs(state2)
// count2 是 Ref<number>，保持响应性
```

## 三、注意事项与常见陷阱

1. **reactive 不能代理基本类型**：`reactive(0)` 报错，基本类型用 ref
2. **解构 reactive 会丢失响应性**：必须使用 toRefs 或 toRef
3. **重新赋值整个对象会丢失代理**：只能修改对象属性，不能替换整个对象
4. **Reactive 对象类型在 TS 中有限制**：不能直接用接口联合类型创建 reactive
5. **Map/Set 的 reactive 需要显式创建**：`reactive(new Map())` 而非 `reactive({})`
