# 列表渲染 v-for

## 一、概念说明

`v-for` 指令基于数组、对象或数字进行列表渲染。语法格式为 `item in items`（或 `item of items`），其中 `items` 是数据源，`item` 是当前迭代项。

```vue
<script setup>
import { ref } from 'vue'

const fruits = ref([
  { id: 1, name: '苹果', price: 5 },
  { id: 2, name: '香蕉', price: 3 },
  { id: 3, name: '橙子', price: 4 }
])
</script>

<template>
  <!-- 遍历数组 -->
  <ul>
    <li v-for="fruit in fruits" :key="fruit.id">
      {{ fruit.name }} - ¥{{ fruit.price }}
    </li>
  </ul>
</template>
```

## 二、具体用法

### 2.1 遍历数组（带索引）

```vue
<template>
  <ul>
    <li v-for="(item, index) in fruits" :key="item.id">
      {{ index + 1 }}. {{ item.name }}
    </li>
  </ul>
</template>
```

### 2.2 遍历对象

```vue
<script setup>
import { reactive } from 'vue'
const user = reactive({
  name: '张三',
  age: 25,
  city: '北京'
})
</script>

<template>
  <!-- value, key, index -->
  <ul>
    <li v-for="(value, key, index) in user" :key="key">
      {{ index }}. {{ key }}: {{ value }}
    </li>
  </ul>
</template>
```

### 2.3 遍历数字

```vue
<template>
  <!-- 从 1 开始到 n -->
  <span v-for="n in 5" :key="n">{{ n }} </span>
  <!-- 输出: 1 2 3 4 5 -->
</template>
```

### 2.4 在 `<template>` 上使用

```vue
<template>
  <template v-for="item in list" :key="item.id">
    <h3>{{ item.title }}</h3>
    <p>{{ item.content }}</p>
    <hr />
  </template>
</template>
```

## 三、注意事项与常见陷阱

- **必须**为每个 `v-for` 添加 `:key` 属性
- `key` 应该是唯一且稳定的标识（优先用 `id`，避免用 `index`）
- 不要在 `v-for` 中使用 `v-if`（Vue 3 中 v-for 优先级更高，但仍然不推荐）
- 使用索引作为 key 会导致列表更新时出现 bug
- 响应式数组的方法（push、pop、splice 等）会触发视图更新
