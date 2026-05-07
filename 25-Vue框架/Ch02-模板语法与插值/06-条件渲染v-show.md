# 条件渲染 v-show

## 一、概念说明

`v-show` 通过 CSS `display` 属性控制元素的显示和隐藏。元素始终保留在 DOM 中，只是切换 `display: none`。相比 `v-if`，`v-show` 有更高的初始渲染开销但切换开销更低。

```vue
<script setup>
import { ref } from 'vue'

const isVisible = ref(true)
</script>

<template>
  <!-- v-show 切换 display 属性 -->
  <div v-show="isVisible">可见内容</div>

  <!-- 始终渲染在 DOM 中，只是隐藏 -->
  <!-- 实际 DOM: <div style="display: none;">可见内容</div> -->
</template>
```

## 二、具体用法

### 2.1 v-show vs v-if 对比

```vue
<script setup>
import { ref } from 'vue'
const show = ref(true)
</script>

<template>
  <!-- v-show: 元素始终在 DOM 中，切换 display -->
  <div v-show="show">v-show 内容</div>

  <!-- v-if: 条件为 false 时从 DOM 移除 -->
  <div v-if="show">v-if 内容</div>
</template>
```

| 特性 | v-show | v-if |
|------|--------|------|
| DOM 操作 | 不移除，切换 display | 条件变化时增删 DOM |
| 初始开销 | 较低（始终渲染） | 较高（按需渲染） |
| 切换开销 | 低（仅 CSS 切换） | 高（DOM 操作） |
| 适用场景 | 频繁切换 | 条件很少改变 |
| `<template>` | 不支持 | 支持 |
| `v-else` | 不支持 | 支持 |

### 2.2 典型使用场景

```vue
<script setup>
import { ref } from 'vue'
const showDropdown = ref(false)
const showModal = ref(false)
</script>

<template>
  <!-- v-show: 频繁切换（如菜单、选项卡） -->
  <div v-show="showDropdown">下拉菜单内容</div>

  <!-- v-if: 不频繁切换（如模态框、权限控制） -->
  <div v-if="showModal">模态框内容</div>
</template>
```

## 三、注意事项与常见陷阱

- `v-show` 不支持 `<template>` 元素
- `v-show` 不支持 `v-else` 和 `v-else-if`
- `v-show` 的元素始终会被渲染并保留在 DOM 中
- `v-show` 的初始渲染开销比 `v-if` 高（即使初始为 false 也会渲染）
- `v-show` 的样式优先级可能被 CSS `!important` 覆盖
