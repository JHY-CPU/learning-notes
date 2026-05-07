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
  <!-- Vue 3 支持多个根节点 -->
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
  <!-- 不需要额外的包裹 div -->
  <label>{{ label }}</label>
  <input v-model="name" />
  <span class="hint">请输入真实姓名</span>
</template>
```

### 2.2 Fragment 与 Attribute 继承

```vue
<script setup>
// 多根节点时，属性不会自动继承
// 需要手动指定继承位置
defineOptions({
  inheritAttrs: false
})
</script>

<template>
  <!-- 显式绑定 $attrs -->
  <header>标题</header>
  <main v-bind="$attrs">内容</main>
  <footer>页脚</footer>
</template>
```

### 2.3 Fragment 的限制

```vue
<template>
  <!-- Fragment 不支持 CSS 样式 -->
  <!-- 以下样式不会生效 -->
  <!--
  <style>
    /* 错误: Fragment 没有对应 DOM 元素 */
    header, main, footer {
      margin: 0;
    }
  </style>
  -->
</template>
```

## 三、注意事项与常见陷阱

- 多根节点时，`$attrs` 不会自动应用到任何元素上
- Fragment 没有真实的 DOM 节点，不能直接添加 class 或 style
- `<Transition>` 组件要求只有一个根元素，多根节点需要用 `<TransitionGroup>`
- 在 DevTools 中，Fragment 不会显示为独立节点
- 某些 CSS 选择器（如 `> *`）可能需要调整
