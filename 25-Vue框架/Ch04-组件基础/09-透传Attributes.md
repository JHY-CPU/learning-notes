# 透传 Attributes

## 一、概念说明

**透传 Attributes**（Fallthrough Attributes）指传递给组件但未在 props 或 emits 中声明的 attribute。这些 attribute 会"透传"到组件的根元素上，包括 `class`、`style`、`id` 等原生属性。

```vue
<!-- 子组件: MyButton.vue -->
<script setup>
defineProps({
  type: { type: String, default: 'button' }
})
</script>

<template>
  <!-- class, style, id 等会透传到此元素 -->
  <button :type="type">
    <slot />
  </button>
</template>

<!-- 父组件 -->
<!-- <MyButton class="primary" id="submit-btn" data-test="btn">
  提交
</MyButton> -->
<!-- 实际渲染: <button type="button" class="primary" id="submit-btn" data-test="btn">提交</button> -->
```

## 二、具体用法

### 2.1 访问透传属性

```vue
<script setup>
import { useAttrs } from 'vue'

const attrs = useAttrs()
// attrs 包含所有未被 props 声明的属性
console.log(attrs.class)
console.log(attrs.id)
</script>

<template>
  <div v-bind="attrs">内容</div>
</template>
```

### 2.2 禁用透传

```vue
<script setup>
defineOptions({
  inheritAttrs: false // 禁用自动透传到根元素
})

const attrs = useAttrs()
</script>

<template>
  <div>
    <input v-bind="attrs" /> <!-- 手动绑定到子元素 -->
  </div>
</template>
```

### 2.3 多根节点组件

```vue
<script setup>
// 多根节点不会自动透传，需要显式绑定
const attrs = useAttrs()
</script>

<template>
  <header>标题</header>
  <main v-bind="attrs">内容</main> <!-- 需要手动绑定 -->
  <footer>页脚</footer>
</template>
```

## 三、注意事项与常见陷阱

- 透传的 class 和 style 会与根元素的 class/style **合并**
- 事件监听器也会透传（如 `@click`）
- `v-model` 不是透传 attribute
- `defineProps` 中声明的属性不会透传
- `inheritAttrs: false` 禁用自动透传后需要手动 `v-bind="$attrs"`
