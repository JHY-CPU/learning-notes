# 属性绑定 v-bind

## 一、概念说明

`v-bind` 指令用于动态绑定 HTML 属性。可以用简写语法 `:` 代替 `v-bind:`。响应式数据变化时，绑定的属性值会自动更新。

```vue
<script setup>
import { ref } from 'vue'

const imageUrl = ref('/images/logo.png')
const linkUrl = ref('https://vuejs.org')
const isDisabled = ref(false)
const inputId = ref('username-input')
</script>

<template>
  <!-- 完整写法 -->
  <img v-bind:src="imageUrl" alt="Logo" />

  <!-- 简写（推荐） -->
  <img :src="imageUrl" alt="Logo" />

  <!-- 动态链接 -->
  <a :href="linkUrl">Vue 官网</a>

  <!-- 布尔属性 -->
  <button :disabled="isDisabled">按钮</button>

  <!-- 动态 id -->
  <input :id="inputId" />
</template>
```

## 二、具体用法

### 2.1 绑定多个属性

```vue
<script setup>
import { ref } from 'vue'
const imgAttrs = ref({
  src: '/photo.jpg',
  alt: '照片',
  width: 200,
  height: 150
})
</script>

<template>
  <!-- 使用 v-bind 绑定对象 -->
  <img v-bind="imgAttrs" />
  <!-- 等价于 -->
  <img :src="imgAttrs.src" :alt="imgAttrs.alt"
       :width="imgAttrs.width" :height="imgAttrs.height" />
</template>
```

### 2.2 动态属性名

```vue
<script setup>
import { ref } from 'vue'
const attributeName = ref('title')
const attributeValue = ref('鼠标悬停提示')
</script>

<template>
  <div :[attributeName]="attributeValue">动态属性</div>
  <!-- 等价于: <div title="鼠标悬停提示"> -->
</template>
```

### 2.3 布尔属性处理

```vue
<script setup>
import { ref } from 'vue'
const isActive = ref(true)
const hasError = ref(false)
</script>

<template>
  <!-- 对于布尔属性，truthy 值会添加属性，falsy 会移除 -->
  <button :disabled="hasError">提交</button>
  <div :hidden="!isActive">内容</div>
</template>
```

## 三、注意事项与常见陷阱

- `:disabled` 对于 `<button>`、`<input>` 等表单元素有效
- `v-bind="objectProps"` 可以一次性绑定多个属性
- 属性名动态绑定时不能有空格和引号
- `null`、`undefined`、`false` 会移除布尔属性（如 `disabled`）
- 对于非布尔属性，`false` 会转为字符串 `"false"`
