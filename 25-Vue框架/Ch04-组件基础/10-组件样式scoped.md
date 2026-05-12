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
/* 穿透 scoped，影响子组件或 v-html 内容 */
.parent :deep(.child-class) {
  color: red;
}

/* 影响第三方组件库的内部样式 */
.el-input :deep(.el-input__inner) {
  border-color: #42b883;
}
</style>
```

### 2.2 插槽选择器 :slotted()

```vue
<style scoped>
/* 影响插槽传入的内容 */
:slotted(.slot-class) {
  font-weight: bold;
}
</style>
```

### 2.3 全局选择器 :global()

```vue
<style scoped>
/* 在 scoped 样式中定义全局样式 */
:global(.global-class) {
  margin: 0;
}

/* 实际上不常用，建议用单独的 <style> 块 */
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

### 2.5 scoped 与全局样式共存

```vue
<!-- 全局样式 -->
<style>
body {
  margin: 0;
  font-family: system-ui, sans-serif;
}
</style>

<!-- 组件样式 -->
<style scoped>
.container {
  padding: 20px;
}
</style>
```

## 三、常见用例

### 3.1 样式穿透场景

| 场景 | 解决方案 |
|------|---------|
| 修改子组件内部样式 | `:deep()` |
| 修改第三方组件库样式 | `:deep()` |
| 影响插槽内容样式 | `:slotted()` |
| 使用 JS 变量控制样式 | `v-bind()` |

## 四、注意事项与常见陷阱

- scoped 样式不能影响子组件的内部元素（需要 `:deep()`）
- `:deep()` 应该尽量少用，保持组件封装性
- scoped 的性能略低于全局 CSS（需要额外的属性匹配）
- 动态 `v-bind()` 在 CSS 中创建 CSS 变量，有轻微性能开销
- class 和 id 选择器比标签选择器在 scoped 中更高效
- 一个组件可以同时有 `<style scoped>` 和 `<style>`（全局）
- 使用 CSS Modules (`<style module>`) 也是样式隔离的替代方案
