# 组件样式 scoped

## 一、概念说明

`scoped` 属性让 `<style>` 中的 CSS 只作用于当前组件。Vue 通过给元素添加唯一的 `data-v-xxx` 属性和 CSS 选择器后缀实现样式隔离。

```vue
<script setup>
import { ref } from 'vue'
const title = ref('Scoped 样式')
</script>

<template>
  <div class="container">
    <h2>{{ title }}</h2>
    <p>这段文字的样式只影响当前组件</p>
  </div>
</template>

<style scoped>
/* 只在当前组件生效 */
.container {
  padding: 20px;
  border: 1px solid #eee;
}

h2 {
  color: #42b883;
}
</style>
```

## 二、具体用法

### 2.1 深度选择器 :deep()

```vue
<style scoped>
/* 穿透 scoped，影响子组件 */
.parent :deep(.child-class) {
  color: red;
}

/* 等价于 Vue 2 的 >>> 或 /deep/ */
</style>
```

### 2.2 插槽选择器 :slotted()

```vue
<style scoped>
/* 影响插槽内容 */
:slotted(.slot-class) {
  font-weight: bold;
}
</style>
```

### 2.3 全局选择器 :global()

```vue
<style scoped>
:global(.global-class) {
  margin: 0;
}
</style>
```

### 2.4 动态 CSS 中的 v-bind

```vue
<script setup>
import { ref } from 'vue'
const color = ref('#42b883')
const size = ref(16)
</script>

<template>
  <p class="dynamic">动态样式文本</p>
</template>

<style scoped>
.dynamic {
  color: v-bind(color);
  font-size: v-bind(size + 'px');
}
</style>
```

## 三、注意事项与常见陷阱

- scoped 样式不能影响子组件的内部元素（需要 `:deep()`）
- `:deep()` 应该尽量少用，保持组件封装性
- scoped 的性能略低于全局 CSS（需要额外的属性匹配）
- 动态 `v-bind()` 在 CSS 中创建 CSS 变量，有轻微性能开销
- class 和 id 选择器比标签选择器在 scoped 中更高效
