# Fragment 片段

## 一、概念说明

Vue 3 支持**多根节点组件**（Fragment）。Vue 2 中组件模板必须有且只有一个根元素，Vue 3 解除了这个限制。Vue 3 的虚拟 DOM 实现了 Fragment 类型，可以包含多个子节点而不需要包裹元素。

```vue
<script setup>
import { ref } from 'vue'

const title = ref('标题')
const content = ref('内容')
</script>

<template>
  <header>{{ title }}</header>
  <main>{{ content }}</main>
  <footer>页脚</footer>
</template>
```

## 二、具体用法

### 2.1 多根节点使用

```vue
<script setup>
import { ref } from 'vue'
const label = ref('姓名')
const name = ref('')
</script>

<template>
  <label>{{ label }}</label>
  <input v-model="name" />
  <span class="hint">请输入真实姓名</span>
</template>
```

### 2.2 Fragment 与 Attribute 继承

```vue
<script setup>
defineOptions({
  inheritAttrs: false
})
</script>

<template>
  <header>标题</header>
  <main v-bind="$attrs">内容</main>
  <footer>页脚</footer>
</template>
```

多根节点时属性不会自动继承，需要通过 `v-bind="$attrs"` 手动指定。

### 2.3 实用场景：列表组件

```vue
<!-- ListItem.vue - 不需要额外包裹 -->
<script setup>
defineProps({
  item: Object
})
</script>

<template>
  <dt>{{ item.term }}</dt>
  <dd>{{ item.detail }}</dd>
</template>
```

```vue
<!-- 使用 -->
<dl>
  <ListItem v-for="item in glossary" :key="item.id" :item="item" />
</dl>
```

### 2.4 Fragment 与样式

```vue
<style scoped>
/* Fragment 没有对应的 DOM 元素 */
/* 以下样式可以直接作用于各根节点 */
header {
  font-size: 1.5rem;
}
main {
  padding: 1rem;
}
footer {
  border-top: 1px solid #eee;
}
</style>
```

## 三、常见用例

### 3.1 何时使用 Fragment

| 场景 | 建议 |
|------|------|
| 组件需要输出相邻元素 | 使用 Fragment（多根节点） |
| 组件需要整体的容器样式 | 使用单根节点 `<div>` |
| `<Transition>` 组件包裹 | 必须单根节点 |
| 表格行组件 `<tr>` | Fragment 避免额外 `<div>` 破坏表格结构 |

### 3.2 迁移自 Vue 2

```vue
<!-- Vue 2：必须有单一根节点 -->
<template>
  <div>
    <header>标题</header>
    <main>内容</main>
  </div>
</template>

<!-- Vue 3：可以去掉多余包裹 -->
<template>
  <header>标题</header>
  <main>内容</main>
</template>
```

## 四、注意事项与常见陷阱

- 多根节点时，`$attrs` 不会自动应用到任何元素上
- Fragment 没有真实的 DOM 节点，不能直接添加 class 或 style
- `<Transition>` 组件要求只有一个根元素，多根节点需要用 `<TransitionGroup>`
- 在 DevTools 中，Fragment 不会显示为独立节点
- 某些 CSS 选择器（如 `> *`）可能需要调整
- Fragment 不影响组件的 props 传递和事件触发
- 在 DOM 模板中，某些 HTML 结构（如 `<table>` 内部）对子元素有严格要求，Fragment 非常有用
