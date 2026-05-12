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

## 四、高级用法

### 4.1 绑定 HTML 内容
```vue
<template>
  <!-- v-html 是 v-bind 的特殊形式 -->
  <div v-html="rawHtml"></div>

  <!-- 动态绑定 innerHTML -->
  <div :innerHTML="content"></div>
</template>
<script setup>
import { ref } from 'vue'
const rawHtml = ref('<strong>加粗文字</strong>')
const content = ref('<em>斜体文字</em>')
</script>
```

### 4.2 条件性绑定属性
```vue
<template>
  <!-- null/undefined/false 会移除属性 -->
  <button :disabled="isSubmitting">
    {{ isSubmitting ? '提交中...' : '提交' }}
  </button>

  <!-- 对象语法：值为 true 的属性被绑定 -->
  <div v-bind="{ id: 'main', 'data-type': type, disabled: isDisabled }"></div>
</template>
```

### 4.3 绑定 style 的多种写法
```vue
<template>
  <!-- 对象语法 -->
  <div :style="{ color: textColor, fontSize: size + 'px' }"></div>

  <!-- 数组语法：多个样式对象 -->
  <div :style="[baseStyles, overrideStyles]"></div>

  <!-- CSS 变量 -->
  <div :style="{ '--primary-color': color }"></div>

  <!-- 自动前缀 -->
  <div :style="{ transform: 'rotate(45deg)' }"></div>
  <!-- Vue 自动添加 -webkit- 前缀 -->
</template>
```

### 4.4 class 的对象与数组组合
```vue
<template>
  <!-- 组合写法 -->
  <div
    class="base-class"
    :class="[
      { active: isActive, disabled: isDisabled },
      dynamicClass,
      conditionalClass ? 'show' : 'hide'
    ]"
  ></div>
</template>
```

## 五、性能注意事项

- 避免在 `:style` 中使用大量内联样式，考虑用 CSS class
- 动态 `:class` 中的对象/数组每次渲染都会创建新引用，可用 computed 缓存
- `v-bind="object"` 会绑定所有属性，确保对象中没有意外属性
