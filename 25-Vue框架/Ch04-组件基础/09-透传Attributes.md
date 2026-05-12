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
  <button :type="type">
    <slot />
  </button>
</template>

<!-- 父组件使用 -->
<!-- <MyButton class="primary" id="submit-btn">提交</MyButton> -->
<!-- 实际渲染: <button type="button" class="primary" id="submit-btn">提交</button> -->
```

## 二、具体用法

### 2.1 访问透传属性 useAttrs

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

### 2.2 禁用自动透传

```vue
<script setup>
defineOptions({
  inheritAttrs: false
})

const attrs = useAttrs()
</script>

<template>
  <div>
    <!-- 手动绑定到子元素而非根元素 -->
    <input v-bind="attrs" />
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
  <main v-bind="attrs">内容</main>
  <footer>页脚</footer>
</template>
```

### 2.4 透传事件监听器

```vue
<script setup>
const attrs = useAttrs()
// 事件监听器也在 attrs 中
// attrs.onClick, attrs.onMouseenter 等
</script>

<template>
  <div v-bind="attrs">
    <!-- 点击事件会透传到父组件 -->
    <slot />
  </div>
</template>
```

## 三、常见用例

### 3.1 封装输入组件

```vue
<!-- MyInput.vue -->
<script setup>
defineOptions({ inheritAttrs: false })

defineProps({
  label: String,
  modelValue: String
})

const emit = defineEmits(['update:modelValue'])
const attrs = useAttrs()
</script>

<template>
  <div class="input-wrapper">
    <label v-if="label">{{ label }}</label>
    <input
      v-bind="attrs"
      :value="modelValue"
      @input="emit('update:modelValue', $event.target.value)"
    />
  </div>
</template>
```

## 四、注意事项与常见陷阱

- 透传的 class 和 style 会与根元素的 class/style **合并**
- 事件监听器也会透传（如 `@click`）
- `v-model` 不是透传 attribute
- `defineProps` 中声明的属性不会透传
- `inheritAttrs: false` 禁用自动透传后需要手动 `v-bind="$attrs"`
- `useAttrs()` 返回的不是响应式的（在模板中使用时是响应式的）
- 多根节点时必须手动指定透传位置，否则 Vue 会警告
