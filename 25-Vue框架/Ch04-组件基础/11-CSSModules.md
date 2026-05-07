# CSS Modules

## 一、概念说明

CSS Modules 提供了比 `scoped` 更强的样式隔离。通过 `module` 属性，Vue 将样式类名转为唯一的哈希值，通过 `$style` 对象在模板和 JS 中访问。CSS Modules 避免了全局类名冲突。

```vue
<script setup>
import { useCssModule } from 'vue'

// 获取 $style 对象
const style = useCssModule()
console.log(style.container) // 唯一的类名如 "container_abc123"
</script>

<template>
  <div :class="$style.container">
    <h2 :class="[$style.title, $style.large]">CSS Modules</h2>
    <p :class="$style.text">样式通过 $style 访问</p>
  </div>
</template>

<style module>
.container {
  padding: 20px;
  border: 1px solid #eee;
}
.title {
  color: #42b883;
}
.text {
  color: #666;
}
.large {
  font-size: 24px;
}
</style>
```

## 二、具体用法

### 2.1 使用 $style

```vue
<template>
  <!-- 绑定 class -->
  <div :class="$style.wrapper">

  <!-- 对象语法 -->
  <p :class="{ [$style.active]: isActive }">内容</p>

  <!-- 数组语法 -->
  <p :class="[$style.text, $style.bold]">内容</p>
</template>
```

### 2.2 自定义 module 名

```vue
<template>
  <div :class="classes.container">内容</div>
</template>

<style module="classes">
.container {
  display: flex;
}
</style>
```

### 2.3 JS 中访问模块类

```vue
<script setup>
import { useCssModule } from 'vue'

const style = useCssModule()

function getClassName() {
  return style.highlight
}
</script>
```

### 2.4 CSS Modules vs Scoped

| 特性 | Scoped CSS | CSS Modules |
|------|-----------|-------------|
| 隔离方式 | data 属性选择器 | 唯一类名哈希 |
| 使用方式 | 普通 CSS | $style 对象 |
| 穿透方式 | :deep() | global() |
| 类型安全 | 无 | 有（TS 支持） |

## 三、注意事项与常见陷阱

- `$style` 只在模板中可用（或通过 `useCssModule()` 获取）
- CSS Modules 的类名在构建时生成，不能动态拼接
- 可以同时使用 `module` 和 `scoped`（不常见）
- CSS Modules 在模板中必须使用绑定语法（`:class`）
- 与 scoped 相比，CSS Modules 更适合组件库开发
