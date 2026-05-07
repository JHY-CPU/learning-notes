# 组件类型化 Props

## 一、概念说明

TypeScript 项目中使用 `defineProps` 的泛型形式声明 props 类型。相比运行时声明，类型声明提供编译时类型检查、IDE 智能提示和自动补全。

```vue
<script setup lang="ts">
// 基于类型的声明
interface Props {
  title: string
  count?: number
  items: Array<{ id: number; label: string }>
  onClick: (id: number) => void
}

const props = defineProps<Props>()

// props 有完整类型推断
console.log(props.title)       // string
console.log(props.count)       // number | undefined
console.log(props.items[0].id) // number
</script>

<template>
  <h2>{{ title }}</h2>
  <ul>
    <li v-for="item in items" :key="item.id">
      {{ item.label }}
    </li>
  </ul>
</template>
```

## 二、具体用法

### 2.1 类型声明 + 默认值

```vue
<script setup lang="ts">
interface Props {
  title: string
  size?: 'small' | 'medium' | 'large'
  count?: number
}

// withDefaults 提供默认值
const props = withDefaults(defineProps<Props>(), {
  size: 'medium',
  count: 0
})
</script>
```

### 2.2 复杂类型

```vue
<script setup lang="ts">
type Status = 'pending' | 'success' | 'error'

interface User {
  id: number
  name: string
  email: string
}

interface Props {
  user: User
  status: Status
  columns?: Array<{ key: string; label: string }>
  formatter?: (value: any) => string
}

const props = defineProps<Props>()
</script>
```

### 2.3 泛型组件

```vue
<script setup lang="ts" generic="T">
// 泛型 props
interface Props {
  items: T[]
  keyExtractor: (item: T) => string | number
  renderItem: (item: T) => string
}

defineProps<Props>()
</script>

<template>
  <ul>
    <li v-for="item in items" :key="keyExtractor(item)">
      {{ renderItem(item) }}
    </li>
  </ul>
</template>
```

## 三、注意事项与常见陷阱

- 泛型形式和运行时形式**不能同时使用**
- `withDefaults` 只能与泛型形式一起使用
- 函数类型的 props 需要完整签名
- 泛型组件需要 Vue 3.3+
- 类型声明只提供编译时检查，运行时验证需要额外配置
