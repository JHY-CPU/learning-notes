# 单文件组件（SFC）

## 一、概念说明

**单文件组件**（Single File Component，简称 SFC）是 Vue 的标志性文件格式，扩展名为 `.vue`。它将组件的**模板**（template）、**逻辑**（script）和**样式**（style）封装在同一个文件中，实现了真正的组件化开发。

SFC 由三部分组成：`<template>` 定义 HTML 模板，`<script>` 定义组件逻辑，`<style>` 定义组件样式。这种结构让组件的职责清晰、易于维护和复用。

```vue
<script setup>
import { ref } from 'vue'

const greeting = ref('你好，Vue 3！')
</script>

<template>
  <div class="greeting">
    <h2>{{ greeting }}</h2>
  </div>
</template>

<style scoped>
.greeting h2 {
  color: #42b883;
  font-weight: bold;
}
</style>
```

## 二、具体用法

### 2.1 `<script setup>` 语法糖

`<script setup>` 是 Vue 3.2+ 引入的编译时语法糖，让 Composition API 使用更简洁。顶层的绑定（变量、函数、import）自动暴露给模板：

```vue
<script setup>
import { ref, computed } from 'vue'

const count = ref(0)
const doubled = computed(() => count.value * 2)
const increment = () => count.value++
</script>

<template>
  <p>{{ count }} x 2 = {{ doubled }}</p>
  <button @click="increment">加一</button>
</template>
```

### 2.2 带 `lang` 属性的 script

使用 TypeScript 时添加 `lang="ts"`：

```vue
<script setup lang="ts">
interface User {
  name: string
  age: number
}

const user: User = { name: '张三', age: 25 }
</script>

<template>
  <p>{{ user.name }}，{{ user.age }}岁</p>
</template>
```

### 2.3 style scoped

`scoped` 让样式只作用于当前组件，避免全局污染：

```vue
<style scoped>
/* 这些样式只在当前组件生效 */
.container {
  padding: 20px;
}
</style>
```

## 三、注意事项与常见陷阱

- 每个 `.vue` 文件最多一个 `<script setup>` 和一个 `<script>`
- `scoped` 样式不会影响子组件内部元素，需要 `:deep()` 穿透
- SFC 需要构建工具（Vite/Webpack）编译，浏览器无法直接运行
- `<template>` 中只能有一个根元素（Vue 3 支持 Fragment，可多根节点）
- 文件名建议使用 PascalCase 命名（如 `MyComponent.vue`）
