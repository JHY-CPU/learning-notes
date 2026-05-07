# v-bind 详解

## 一、概念说明
`v-bind` 用于**动态绑定一个或多个 attribute**到表达式。它是 Vue 中最常用的指令之一，缩写为 `:`。

## 二、具体用法

### 2.1 绑定单个 attribute
```vue
<template>
  <!-- 完整写法 -->
  <img v-bind:src="imageUrl" v-bind:alt="imageAlt" />

  <!-- 缩写 -->
  <img :src="imageUrl" :alt="imageAlt" />
</template>
<script setup>
import { ref } from 'vue'
const imageUrl = ref('/logo.png')
const imageAlt = ref('Logo')
</script>
```

### 2.2 动态 attribute 名
```vue
<template>
  <div :[attributeName]="value">动态属性</div>
  <!-- attributeName = 'id' 时等价于 :id="value" -->
</template>
<script setup>
import { ref } from 'vue'
const attributeName = ref('id')
const value = ref('myDiv')
</script>
```

### 2.3 绑定对象（批量绑定）
```vue
<template>
  <div v-bind="attrs">批量绑定</div>
  <!-- 等价于 :id="attrs.id" :class="attrs.class" ... -->
</template>
<script setup>
const attrs = {
  id: 'container',
  class: 'wrapper',
  'data-type': 'main'
}
</script>
```

### 2.4 绑定 class 和 style
```vue
<template>
  <!-- 对象语法 -->
  <div :class="{ active: isActive, 'text-bold': isBold }"></div>

  <!-- 数组语法 -->
  <div :class="[activeClass, errorClass]"></div>

  <!-- style 对象 -->
  <div :style="{ color: textColor, fontSize: size + 'px' }"></div>
</template>
<script setup>
import { ref } from 'vue'
const isActive = ref(true)
const isBold = ref(false)
</script>
```

## 三、注意事项与常见陷阱
- `v-bind` 的值是 JavaScript 表达式，不是字符串
- 布尔 attribute（如 `disabled`）：值为 `true` 时添加，`false` 时移除
- `null`、`undefined`、`false` 会移除 attribute
- class 和 style 有特殊的合并行为，不会覆盖原有值
