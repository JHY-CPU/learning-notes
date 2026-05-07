# 模板中的 JavaScript

## 一、概念说明

Vue 模板支持在插值 `{{ }}` 和指令中使用 JavaScript 表达式。模板中的表达式被限制在当前组件实例的作用域内，可以访问组件的 data、computed、methods 以及全局白名单对象。

```vue
<script setup>
import { ref } from 'vue'
const number = ref(42)
const items = ref([3, 1, 4, 1, 5])
const message = ref('Hello Vue')
</script>

<template>
  <!-- 算术运算 -->
  <p>{{ number + 1 }}</p>

  <!-- 三元表达式 -->
  <p>{{ number > 0 ? '正数' : '非正数' }}</p>

  <!-- 方法调用 -->
  <p>{{ message.toUpperCase() }}</p>

  <!-- 数组方法 -->
  <p>{{ items.sort().join(', ') }}</p>

  <!-- 全局对象 -->
  <p>{{ Math.round(number * 100) / 100 }}</p>
</template>
```

## 二、具体用法

### 2.1 可访问的全局对象

```vue
<template>
  <!-- 白名单中的全局对象可直接使用 -->
  <p>{{ Math.PI }}</p>
  <p>{{ Date.now() }}</p>
  <p>{{ parseInt('42') }}</p>
  <p>{{ JSON.stringify({ key: 'value' }) }}</p>
</template>
```

### 2.2 不可使用的语句

```vue
<template>
  <!-- 错误: 这不是表达式 -->
  <!-- {{ if (ok) { return message } }} -->

  <!-- 正确: 使用三元表达式 -->
  <p>{{ ok ? message : '' }}</p>

  <!-- 错误: 赋值语句 -->
  <!-- {{ a = 1 }} -->

  <!-- 错误: 链式赋值 -->
  <!-- {{ a = b = c }} -->
</template>
```

### 2.3 复杂逻辑用 computed

```vue
<script setup>
import { ref, computed } from 'vue'
const items = ref([1, 2, 3, 4, 5])

// 不推荐: 在模板中写复杂表达式
// <p>{{ items.filter(i => i % 2 === 0).map(i => i * 2).reduce((a, b) => a + b, 0) }}</p>

// 推荐: 用 computed
const evenDoubledSum = computed(() =>
  items.value.filter(i => i % 2 === 0).map(i => i * 2).reduce((a, b) => a + b, 0)
)
</script>

<template>
  <p>{{ evenDoubledSum }}</p>
</template>
```

## 三、注意事项与常见陷阱

- 模板表达式只能访问组件实例和白名单全局对象
- 不能访问用户定义的全局变量（除非通过 `app.config.globalProperties`）
- 模板中对 ref 自动解包（不需要 `.value`）
- 不要在模板中产生副作用（修改数据、触发异步操作）
- 模板中的表达式会在每次重新渲染时执行
