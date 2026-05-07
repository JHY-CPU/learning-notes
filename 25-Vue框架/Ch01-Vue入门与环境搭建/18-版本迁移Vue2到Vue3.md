# 版本迁移：Vue 2 到 Vue 3

## 一、概念说明

从 Vue 2 迁移到 Vue 3 涉及多个**破坏性变更**（breaking changes）。主要变化包括：全局 API 改为应用实例 API、生命周期钩子更名、v-model 语法变化、事件总线移除等。Vue 官方提供了 `@vue/compat` 构建用于渐进式迁移。

```vue
<script setup>
// Vue 3 迁移后的变化示例
import { createApp } from 'vue'
import App from './App.vue'

// Vue 2: Vue.use(router)
// Vue 3:
const app = createApp(App)
app.use(router)

// Vue 2: Vue.component('MyComp', ...)
// Vue 3: app.component('MyComp', ...)
</script>
```

## 二、具体用法

### 2.1 主要破坏性变更

```vue
<script>
// Vue 2 → Vue 3 生命周期变化
export default {
  // beforeDestroy → beforeUnmount
  // destroyed → unmounted
  beforeUnmount() {
    console.log('Vue 2 中是 beforeDestroy')
  },

  // v-model 变化
  // Vue 2: value + input
  // Vue 3: modelValue + update:modelValue
  props: ['modelValue'],
  emits: ['update:modelValue']
}
</script>

<template>
  <!-- Vue 2: $listeners 已移除，合并到 $attrs -->
  <!-- Vue 2: filters 已移除，用 computed 或方法替代 -->
  <p>{{ formatDate(date) }}</p>
</template>
```

### 2.2 迁移工具

```bash
# 使用兼容构建逐步迁移
pnpm add @vue/compat@3

# 运行官方迁移转换工具
npx @vue/cli-service build --mode production
```

### 2.3 关键 API 变化

| Vue 2 | Vue 3 |
|-------|-------|
| `Vue.use()` | `app.use()` |
| `Vue.component()` | `app.component()` |
| `Vue.directive()` | `app.directive()` |
| `beforeDestroy` | `beforeUnmount` |
| `destroyed` | `unmounted` |
| `$on` / `$off` / `$emit` (事件总线) | mitt 或其他库 |
| `filters` | `computed` 或方法 |
| `v-model` / `.sync` | `v-model:propName` |

## 三、注意事项与常见陷阱

- Vue 2 的 `$listeners` 在 Vue 3 中已合并到 `$attrs`
- `.sync` 修饰符被 `v-model:propName` 替代
- Vue 3 移除了事件总线，推荐使用 mitt 或 Pinia
- `filter` 语法已移除，在 computed 或方法中格式化
- template 中的 `$listeners` 引用需要移除
