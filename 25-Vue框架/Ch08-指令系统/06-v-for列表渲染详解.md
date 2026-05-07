# v-for 列表渲染详解

## 一、概念说明
`v-for` 用于基于一个数组或对象**渲染列表**。它是 Vue 中处理列表数据的核心指令，支持遍历数组、对象和数字范围。

## 二、具体用法

### 2.1 遍历数组
```vue
<template>
  <!-- 基本语法 -->
  <li v-for="item in items" :key="item.id">
    {{ item.name }}
  </li>

  <!-- 带索引 -->
  <li v-for="(item, index) in items" :key="item.id">
    {{ index + 1 }}. {{ item.name }}
  </li>
</template>
<script setup>
import { ref } from 'vue'
const items = ref([
  { id: 1, name: '张三' },
  { id: 2, name: '李四' }
])
</script>
```

### 2.2 遍历对象
```vue
<template>
  <!-- 遍历对象的属性 -->
  <div v-for="(value, key, index) in user" :key="key">
    {{ index }}. {{ key }}: {{ value }}
  </div>
</template>
<script setup>
import { reactive } from 'vue'
const user = reactive({
  name: '张三',
  age: 25,
  city: '北京'
})
</script>
```

### 2.3 遍历数字和字符串
```vue
<template>
  <!-- 1 到 10 -->
  <span v-for="n in 10" :key="n">{{ n }} </span>

  <!-- 字符串 -->
  <span v-for="char in 'hello'" :key="char">{{ char }}</span>
</template>
```

### 2.4 使用解构
```vue
<template>
  <li v-for="{ id, name, age } in users" :key="id">
    {{ name }} - {{ age }}岁
  </li>
</template>
```

### 2.5 v-for 与 v-if 同时使用
```vue
<!-- ❌ 不要同时使用（v-for 优先级更高） -->
<li v-for="item in items" v-if="item.active" :key="item.id">

<!-- ✅ 使用 computed 或 template -->
<li v-for="item in activeItems" :key="item.id">

<!-- ✅ 或用 template 包裹 -->
<template v-for="item in items" :key="item.id">
  <li v-if="item.active">{{ item.name }}</li>
</template>
```

## 三、注意事项与常见陷阱
- **必须提供唯一的 `:key`**，不要用 index 作为 key（除非列表只增不减）
- 不要在 v-for 中同时使用 v-if
- 数组变更方法（push/pop/splice）会触发更新，替换数组也会触发
- 不能检测通过索引设置项的变化：`arr[0] = newVal` 不触发更新
