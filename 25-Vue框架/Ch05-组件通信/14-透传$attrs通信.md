# 透传 $attrs 通信

## 一、概念说明
`$attrs` 包含了父组件传递给子组件的**所有未被 props 声明接收的属性**。通过 `$attrs` 可以将父组件的属性透传到子组件的根元素或任意子元素上，适合封装高阶包装组件。

## 二、具体用法

### 2.1 基本用法
```vue
<!-- 父组件 -->
<MyInput placeholder="请输入" class="large" maxlength="20" />
```

```vue
<!-- MyInput.vue: 只声明了 modelValue prop -->
<template>
  <!-- $attrs 会自动应用到根元素（单根节点时） -->
  <input :value="modelValue" @input="$emit('update:modelValue', $event.target.value)" />
</template>
<script setup>
defineProps<{ modelValue: string }>()
defineEmits(['update:modelValue'])
</script>
<!-- 渲染结果: <input placeholder="请输入" class="large" maxlength="20"> -->
```

### 2.2 禁用自动继承
```vue
<script setup>
defineOptions({ inheritAttrs: false })
</script>

<template>
  <div class="wrapper">
    <!-- 手动将 $attrs 应用到内部元素 -->
    <input v-bind="$attrs" />
  </div>
</template>
```

### 2.3 属性与事件透传
```vue
<script setup>
const attrs = useAttrs()
console.log(attrs) // { placeholder: '请输入', onClick: fn, ... }
</script>
```

## 三、注意事项与常见陷阱
- `$attrs` 包含属性**和**事件监听器（class 和 style 除外）
- 多根节点组件不会自动继承，需手动 `v-bind="$attrs"`
- 使用 `defineOptions({ inheritAttrs: false })` 可禁用自动继承
- 不要在 `$attrs` 中包含 props 已声明的属性，会产生重复
