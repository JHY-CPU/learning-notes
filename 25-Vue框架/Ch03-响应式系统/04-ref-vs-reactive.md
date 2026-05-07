# ref vs reactive 对比

## 一、概念说明

`ref` 和 `reactive` 是 Vue 3 中创建响应式数据的两种方式。`ref` 适用于任何类型，`reactive` 仅适用于对象类型。两者的核心区别在于访问方式和使用场景。

```vue
<script setup>
import { ref, reactive } from 'vue'

// ref: 需要 .value
const count = ref(0)
count.value++

// reactive: 直接访问
const state = reactive({ count: 0 })
state.count++
</script>

<template>
  <!-- 模板中都自动解包 -->
  <p>{{ count }}</p>
  <p>{{ state.count }}</p>
</template>
```

## 二、具体用法

### 2.1 使用场景对比

```vue
<script setup>
import { ref, reactive } from 'vue'

// ref: 基本类型
const name = ref('Vue')
const age = ref(25)
const isActive = ref(true)

// ref: 也可以用于对象（整体替换方便）
const user = ref({ name: '张三', age: 25 })

// reactive: 多属性表单状态
const form = reactive({
  username: '',
  email: '',
  password: '',
  confirmPassword: ''
})
</script>
```

### 2.2 选择策略

| 场景 | 推荐 | 原因 |
|------|------|------|
| 基本类型 | `ref()` | reactive 不支持 |
| 表单状态 | `reactive()` | 多属性组织更自然 |
| 需要整体替换 | `ref()` | ref.value = newObj |
| 解构使用 | `ref()` | 无响应式丢失问题 |
| 函数返回值 | `ref()` | 保持响应式 |

### 2.3 实际项目中的风格

```vue
<script setup>
import { ref, reactive } from 'vue'

// 风格1: 全用 ref（推荐，简单一致）
const count = ref(0)
const user = ref({ name: '张三' })
const increment = () => count.value++

// 风格2: 对象用 reactive
const state = reactive({
  count: 0,
  user: { name: '张三' }
})
const increment2 = () => state.count++
</script>
```

## 三、注意事项与常见陷阱

- reactive 不能用于基本类型，ref 可以
- reactive 解构会丢失响应式，ref 不会（自动解包）
- 不要对同一个对象同时使用 ref 和 reactive
- Vue 官方没有强制推荐，团队应统一风格
- reactive 重新赋值会丢失响应式（`state = newObj` 错误）
