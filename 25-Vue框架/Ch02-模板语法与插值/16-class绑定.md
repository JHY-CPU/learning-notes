# class 绑定

## 一、概念说明

Vue 提供了强大的 class 绑定能力。可以使用对象语法或数组语法动态绑定 CSS class。可以与静态 class 共存，Vue 会自动合并。

```vue
<script setup>
import { ref } from 'vue'

const isActive = ref(true)
const hasError = ref(false)
</script>

<template>
  <!-- 对象语法: { className: condition } -->
  <div :class="{ active: isActive, error: hasError }">
    动态 class
  </div>
  <!-- 当 isActive=true, hasError=false 时: class="active" -->
</template>
```

## 二、具体用法

### 2.1 对象语法

```vue
<script setup>
import { ref, computed } from 'vue'

const isActive = ref(true)
const isError = ref(false)
const status = ref('success')

// 计算属性返回 class 对象
const classObject = computed(() => ({
  active: isActive.value,
  'text-danger': isError.value,
  [`status-${status.value}`]: true
}))
</script>

<template>
  <!-- 内联对象 -->
  <div :class="{ active: isActive, 'text-danger': isError }"></div>

  <!-- 绑定计算属性 -->
  <div :class="classObject"></div>

  <!-- 绑定数组 -->
  <div :class="[isActive ? 'active' : '', isError ? 'error' : '']"></div>
</template>
```

### 2.2 数组语法

```vue
<script setup>
import { ref } from 'vue'
const activeClass = ref('active')
const errorClass = ref('text-danger')
const isActive = ref(true)
</script>

<template>
  <!-- 数组语法 -->
  <div :class="[activeClass, errorClass]"></div>

  <!-- 三元表达式 -->
  <div :class="[isActive ? activeClass : '']"></div>

  <!-- 数组中嵌套对象 -->
  <div :class="[{ active: isActive }, errorClass]"></div>

  <!-- 与静态 class 共存 -->
  <div class="static" :class="{ active: isActive }"></div>
</template>
```

### 2.3 组件上的 class

```vue
<!-- 父组件 -->
<template>
  <!-- 子组件的根元素会接收父组件的 class -->
  <MyComponent class="custom-class" :class="{ active: isActive }" />
</template>
```

## 三、注意事项与常见陷阱

- 对象语法中，key 是 class 名，value 是布尔条件
- class 名中有连字符时需要引号包裹（如 `'text-danger'`）
- 数组语法中可以混合使用字符串、对象和三元表达式
- Vue 自动合并静态 class 和动态 class
- 组件的 class 默认添加到根元素上（除非根元素有多个且禁用 attribute 继承）
