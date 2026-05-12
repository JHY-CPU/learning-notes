# SFC CSS 特性

## 一、概念说明

Vue SFC 的 `<style>` 块支持许多增强特性：`scoped` 样式隔离、`module` CSS Modules、`v-bind()` 在 CSS 中使用 JS 变量、`lang` 预处理器支持等。

```vue
<script setup>
import { ref } from 'vue'

const color = ref('#42b883')
const size = ref(16)
</script>

<template>
  <p class="dynamic-text">动态样式</p>
</template>

<style scoped>
.dynamic-text {
  color: v-bind(color);
  font-size: v-bind(size + 'px');
}
</style>
```

## 二、具体用法

### 2.1 v-bind in CSS

```vue
<script setup>
import { ref, reactive } from 'vue'

const theme = reactive({
  primary: '#42b883',
  fontSize: 14
})
</script>

<style scoped>
.button {
  background-color: v-bind('theme.primary');
  font-size: v-bind('theme.fontSize + "px"');
}
</style>
```

### 2.2 CSS Modules

```vue
<template>
  <p :class="$style.text">CSS Modules 示例</p>
  <p :class="classes.active">活动状态</p>
</template>

<style module>
.text {
  color: #42b883;
  font-size: 14px;
}
.active {
  font-weight: bold;
}
</style>

<!-- 命名模块 -->
<style module="classes">
.active {
  color: red;
}
</style>
```

### 2.3 预处理器支持

```vue
<style lang="scss" scoped>
$primary-color: #42b883;

.container {
  .title {
    color: $primary-color;
    &:hover {
      opacity: 0.8;
    }
  }
}
</style>
```

```bash
# 安装预处理器
pnpm add -D sass     # SCSS
pnpm add -D less     # Less
pnpm add -D stylus   # Stylus
```

### 2.4 CSS 变量注入

```vue
<script setup>
import { ref } from 'vue'
const themeColor = ref('#42b883')
</script>

<template>
  <div :style="{ '--theme-color': themeColor }">
    <p class="themed">使用 CSS 变量</p>
  </div>
</template>

<style scoped>
.themed {
  color: var(--theme-color);
}
</style>
```

### 2.5 多个 style 块

```vue
<!-- 全局基础样式 -->
<style>
:root {
  --primary: #42b883;
  --text: #333;
}
</style>

<!-- 组件 scoped 样式 -->
<style scoped>
.title {
  color: var(--primary);
}
</style>
```

## 三、常见用例

| 特性 | 场景 |
|------|------|
| `scoped` | 样式隔离 |
| `:deep()` | 修改子组件/第三方库样式 |
| `v-bind()` | JS 变量控制 CSS |
| `module` | 类型安全的样式隔离 |
| `lang="scss"` | 使用预处理器 |

## 四、注意事项与常见陷阱

- `v-bind()` 在 CSS 中生成 CSS 变量，有轻微性能开销
- `v-bind()` 中的值变化会触发样式重新计算
- 预处理器（sass/less）需要单独安装依赖
- `<style scoped>` 和 `<style module>` 可以同时使用
- `v-bind()` 在 CSS 中的表达式会被限制为单个表达式
- CSS Modules 的类名在 JS 中通过 `$style.xxx` 访问
- `v-bind()` 支持响应式数据，数据变化时样式会自动更新
