# Props 父传子

## 一、概念说明
Props（属性）是父组件向子组件传递数据的主要方式。Vue 中 props 遵循**单向数据流**原则：父组件的数据变化会流向子组件，但子组件不应直接修改 props 的值。

## 二、具体用法

### 2.1 基本传递
```vue
<!-- 父组件 -->
<template>
  <UserCard name="张三" :age="25" :active="isActive" />
</template>
<script setup>
import { ref } from 'vue'
import UserCard from './UserCard.vue'
const isActive = ref(true)
</script>
```

```vue
<!-- 子组件 UserCard.vue -->
<template>
  <div>{{ name }} - {{ age }}岁</div>
</template>
<script setup>
defineProps({
  name: { type: String, required: true },
  age: { type: Number, default: 0 },
  active: Boolean
})
</script>
```

### 2.2 运行时声明 vs 类型声明
```vue
<script setup>
// 运行时声明
const props = defineProps({
  title: String,
  count: { type: Number, default: 0 }
})

// 基于类型的声明（TypeScript）
const props = defineProps<{
  title: string
  count?: number
}>()
</script>
```

### 2.3 单向数据流规则
```vue
<script setup>
const props = defineProps<{ msg: string }>()

// 错误！不要直接修改 props
// props.msg = '新值'

// 正确做法：用 props 初始化本地数据
import { ref } from 'vue'
const localMsg = ref(props.msg)
</script>
```

### 2.4 传递非原始类型
```vue
<!-- 父组件传递对象和数组 -->
<ListView :items="productList" :config="{ showPrice: true }" />
```

## 三、注意事项与常见陷阱
- props 是只读的，子组件中不应直接修改
- 传递对象/数组时，子组件修改其属性不会报错但会导致数据流混乱
- 使用 `defineProps` 时不需要导入，它是编译器宏
- 数字类型需要使用 `:age="25"` 而非 `age="25"`（后者是字符串）
