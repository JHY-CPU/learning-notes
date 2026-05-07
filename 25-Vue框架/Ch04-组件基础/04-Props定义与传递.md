# Props 定义与传递

## 一、概念说明

Props 是父组件向子组件传递数据的机制。子组件通过 `defineProps` 声明接收的 props，父组件在模板中通过属性绑定传递数据。Props 是**单向数据流**：父组件更新会同步到子组件，但子组件不应修改 props。

```vue
<!-- 子组件: UserCard.vue -->
<script setup>
defineProps({
  name: { type: String, required: true },
  age: { type: Number, default: 18 },
  avatar: { type: String, default: '' }
})
</script>

<template>
  <div class="card">
    <img :src="avatar" :alt="name" />
    <h3>{{ name }}</h3>
    <p>{{ age }} 岁</p>
  </div>
</template>

<!-- 父组件使用 -->
<!-- <UserCard name="张三" :age="25" avatar="/avatar.jpg" /> -->
```

## 二、具体用法

### 2.1 运行时声明

```vue
<script setup>
defineProps({
  // 基础类型
  title: String,
  count: Number,

  // 带默认值
  message: {
    type: String,
    default: '默认消息'
  },

  // 必填
  id: {
    type: [String, Number],
    required: true
  },

  // 对象/数组默认值用函数返回
  list: {
    type: Array,
    default: () => []
  }
})
</script>
```

### 2.2 TypeScript 类型声明

```vue
<script setup lang="ts">
// 基于类型的声明
interface Props {
  title: string
  count?: number       // 可选
  items: string[]      // 必填
  callback: () => void
}

const props = defineProps<Props>()

// 带默认值
const props2 = withDefaults(defineProps<Props>(), {
  count: 0
})
</script>
```

### 2.3 父组件传递 props

```vue
<!-- 父组件 -->
<script setup>
import UserCard from './UserCard.vue'
const user = { name: '张三', age: 25 }
</script>

<template>
  <!-- 静态传值 -->
  <UserCard name="李四" :age="30" />

  <!-- 动态传值 -->
  <UserCard :name="user.name" :age="user.age" />

  <!-- v-bind 传对象 -->
  <UserCard v-bind="user" />
  <!-- 等价于: <UserCard :name="user.name" :age="user.age" /> -->
</template>
```

## 三、注意事项与常见陷阱

- Props 是只读的，不要在子组件中修改
- 对象和数组的默认值必须用工厂函数返回
- 未声明的 props 会作为 attribute 传递到根元素
- Boolean 类型 props 传空字符串被视为 `true`
- `defineProps` 是编译宏，不能在非 `<script setup>` 中使用
