# Props类型定义

## 一、概念说明

Vue 3.3+ 支持 `defineProps` 的泛型语法，直接用 TypeScript 接口定义 Props 类型，比运行时声明更简洁且类型安全。泛型语法自动推断 required/optional，不需要额外写 `withDefaults` 来定义默认值以外的选项。

## 二、具体用法

### 基本泛型语法

```vue
<script setup lang="ts">
// 方式1：直接定义接口
interface Props {
  title: string
  count: number
  visible?: boolean   // ? 表示可选
}

const props = defineProps<Props>()
// props.title   → string（必填）
// props.count   → number（必填）
// props.visible → boolean | undefined（可选）

// 默认值通过 withDefaults 设置
const propsWithDefaults = withDefaults(defineProps<Props>(), {
  visible: true,
  count: 0
})
// visible 默认为 true
</script>

<template>
  <div>
    <h2>{{ title }}</h2>
    <p>数量: {{ count }}</p>
    <div v-if="visible">可见内容</div>
  </div>
</template>
```

### 复杂 Props 类型

```vue
<script setup lang="ts">
interface User {
  id: number
  name: string
  email: string
}

interface Props {
  // 基本类型
  title: string
  width: number
  disabled?: boolean

  // 对象类型
  user: User

  // 数组类型
  tags: string[]
  items: Array<{ id: number; label: string }>

  // 函数类型
  onSubmit?: (data: User) => void
  formatter?: (value: number) => string

  // 联合类型
  status: 'active' | 'inactive' | 'pending'
  size: 'small' | 'medium' | 'large'
}

const props = defineProps<Props>()

// 使用 props
const formattedWidth = computed(() => {
  return props.formatter ? props.formatter(props.width) : `${props.width}px`
})
</script>
```

### 泛型 Props

```vue
<!-- GenericList.vue -->
<script setup lang="ts">
// 组件本身支持泛型
interface Props<T> {
  items: T[]
  keyExtractor: (item: T) => string | number
  renderItem: (item: T) => string
}

// Vue 3.3+ 支持泛型 defineProps
defineProps<Props<{ id: number; name: string }>>()
</script>

<template>
  <ul>
    <li v-for="item in items" :key="keyExtractor(item)">
      {{ renderItem(item) }}
    </li>
  </ul>
</template>
```

### 使用示例

```vue
<!-- 父组件 -->
<script setup lang="ts">
import UserCard from './UserCard.vue'

const handleUserSubmit = (data: { id: number; name: string; email: string }) => {
  console.log('提交用户:', data.name)
  // 输出：提交用户: 张三
}
</script>

<template>
  <UserCard
    title="用户信息"
    :width="300"
    :user="{ id: 1, name: '张三', email: 'zhang@example.com' }"
    :tags="['管理员', '活跃']"
    status="active"
    size="medium"
    :on-submit="handleUserSubmit"
  />
</template>
```

## 三、注意事项与常见陷阱

1. **泛型语法需要 Vue 3.3+**：旧版本只能使用运行时 `defineProps({ ... })` 声明
2. **不能同时使用泛型和运行时声明**：二选一，推荐泛型语法
3. **withDefaults 只接受箭头函数返回值**：`withDefaults(defineProps<Props>(), { key: () => value })`
4. **Props 解构会丢失响应性**：使用 `toRefs(props)` 保持响应性
5. **可选 Props 类型包含 undefined**：`visible?: boolean` 实际是 `boolean | undefined`
