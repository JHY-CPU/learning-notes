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

### 2.5 v-for 与 v-if 配合

```vue
<script setup>
import { ref } from 'vue'
const todos = ref([
  { id: 1, text: '学习 Vue', done: true },
  { id: 2, text: '写项目', done: false },
  { id: 3, text: '看文档', done: true }
])
</script>

<template>
  <!-- ✅ 正确：用 computed 过滤 -->
  <li v-for="todo in doneTodos" :key="todo.id">{{ todo.text }}</li>
</template>

<script>
import { computed } from 'vue'
const doneTodos = computed(() => todos.value.filter(t => t.done))
</script>
```

### 2.6 列表排序

```vue
<script setup>
import { ref, computed } from 'vue'
const items = ref([
  { id: 1, name: 'Banana', price: 3 },
  { id: 2, name: 'Apple', price: 5 },
  { id: 3, name: 'Cherry', price: 8 }
])

const sortedItems = computed(() =>
  [...items.value].sort((a, b) => a.price - b.price)
)
</script>

<template>
  <li v-for="item in sortedItems" :key="item.id">
    {{ item.name }} - ¥{{ item.price }}
  </li>
</template>
```

## 三、常见用例

### 3.1 key 的重要性

```
key 的作用：帮助 Vue 识别每个节点的身份，从而复用和重新排序现有元素。

没有 key 或使用 index 作为 key 时：
  [A, B, C] → 删除 B → [A, C]
  Vue 可能错误地复用节点，导致状态混乱

使用唯一 id 作为 key 时：
  [A, B, C] → 删除 B → [A, C]
  Vue 正确识别 B 被移除，A 和 C 保持不变
```

## 四、注意事项与常见陷阱

- **必须**为每个 `v-for` 添加 `:key` 属性
- `key` 应该是唯一且稳定的标识（优先用 `id`，避免用 `index`）
- 不要在 `v-for` 中使用 `v-if`（Vue 3 中 v-for 优先级更高，但仍然不推荐，改用 computed 过滤）
- 使用索引作为 key 会导致列表更新时出现 bug
- 响应式数组的方法（push、pop、splice 等）会触发视图更新
- `v-for` 遍历对象时顺序基于 `Object.keys()` 的枚举顺序
- 使用 `v-for` with `v-model` 时，确保每个输入框绑定到独立的数据
