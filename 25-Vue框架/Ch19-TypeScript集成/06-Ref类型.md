# Ref类型

## 一、概念说明

`ref()` 返回 `Ref<T>` 类型对象，`.value` 属性存储实际值。Vue 的类型系统自动推断简单类型的 ref 类型，复杂类型需要显式指定泛型。理解 Ref 类型对正确使用组合式 API 至关重要。

## 二、具体用法

### 自动类型推断

```ts
import { ref } from 'vue'

// 基本类型自动推断
const count = ref(0)
// count 类型: Ref<number>
// count.value 类型: number

const message = ref('你好')
// message 类型: Ref<string>

const isActive = ref(false)
// isActive 类型: Ref<boolean>
```

### 显式泛型指定

```ts
import { ref, type Ref } from 'vue'

// 联合类型需要显式指定
const status = ref<'loading' | 'success' | 'error'>('loading')
// status 类型: Ref<'loading' | 'success' | 'error'>

// 可空类型
const user = ref<{ name: string } | null>(null)
// user 类型: Ref<{ name: string } | null>

// 初始值为 null 的 ref 必须指定类型
const inputEl = ref<HTMLInputElement | null>(null)
// inputEl 类型: Ref<HTMLInputElement | null>

// 数组 ref
const items = ref<Array<{ id: number; text: string }>>([])
// items 类型: Ref<Array<{ id: number; text: string }>>
```

### Ref 类型在组件中的应用

```vue
<script setup lang="ts">
import { ref, type Ref } from 'vue'

// 基本 ref
const count = ref(0)
const increment = () => count.value++
// count.value++ 的类型安全：只能加 number

// 复杂类型 ref
interface Todo {
  id: number
  text: string
  done: boolean
}

const todos = ref<Todo[]>([])
const newTodo = ref('')

function addTodo() {
  todos.value.push({
    id: Date.now(),
    text: newTodo.value,
    done: false
  })
  newTodo.value = ''
}

// 函数 ref
const callback = ref<((msg: string) => void) | null>(null)

// 模板引用
const inputRef = ref<HTMLInputElement | null>(null)
function focusInput() {
  inputRef.value?.focus()
  // 类型安全：TypeScript 知道 inputRef.value 有 focus 方法
}
</script>

<template>
  <div>
    <input ref="inputRef" v-model="newTodo" @keyup.enter="addTodo" />
    <button @click="focusInput">聚焦输入框</button>
    <p>计数: {{ count }} <button @click="increment">+1</button></p>
    <ul>
      <li v-for="todo in todos" :key="todo.id" :class="{ done: todo.done }">
        {{ todo.text }}
      </li>
    </ul>
  </div>
</template>
```

### 自定义 Ref

```ts
import { customRef } from 'vue'

// 带防抖的 ref
function useDebouncedRef<T>(value: T, delay = 200) {
  let timeout: ReturnType<typeof setTimeout>
  return customRef((track, trigger) => ({
    get() {
      track()
      return value
    },
    set(newVal: T) {
      clearTimeout(timeout)
      timeout = setTimeout(() => {
        value = newVal
        trigger()
      }, delay)
    }
  }))
}

const text = useDebouncedRef('hello')
// text 类型: Ref<string>
// 输入停止 200ms 后才更新值
```

## 三、注意事项与常见陷阱

1. **模板中自动解包，JS 中需要 .value**：`<p>{{ count }}</p>` 正确，JS 中 `count + 1` 为 NaN
2. **初始值为 null 必须显式指定类型**：`ref(null)` 类型为 `Ref<null>`，无法赋值其他类型
3. **Ref 解构会失去响应性**：`const { value } = count` 后 value 不再响应
4. **shallowRef 不深度追踪**：适用于大对象或外部状态管理
5. **ref 在 reactive 中自动解包**：`reactive({ count: ref(0) })` 中 count 不需要 .value
