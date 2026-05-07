# style 绑定

## 一、概念说明

Vue 允许使用对象语法或数组语法动态绑定内联样式（inline style）。`:style` 的语法与 CSS 属性基本一致，但可以使用 camelCase 或 kebab-case。

```vue
<script setup>
import { ref } from 'vue'

const textColor = ref('red')
const fontSize = ref(16)
const isActive = ref(true)
</script>

<template>
  <!-- 对象语法 -->
  <p :style="{ color: textColor, fontSize: fontSize + 'px' }">
    动态样式文本
  </p>

  <!-- 直接绑定样式对象 -->
  <p :style="styleObject">绑定样式对象</p>
</template>
```

## 二、具体用法

### 2.1 对象语法

```vue
<script setup>
import { ref, computed } from 'vue'

const color = ref('#42b883')
const size = ref(20)
const weight = ref('bold')

// 样式对象
const styleObject = computed(() => ({
  color: color.value,
  fontSize: `${size.value}px`,
  fontWeight: weight.value
}))
</script>

<template>
  <!-- 内联对象 -->
  <p :style="{ color, fontSize: size + 'px', 'font-weight': weight }">
    内联样式
  </p>

  <!-- 计算属性 -->
  <p :style="styleObject">计算属性样式</p>
</template>
```

### 2.2 数组语法

```vue
<script setup>
import { ref } from 'vue'
const baseStyles = ref({ color: 'blue', fontSize: '16px' })
const overrideStyles = ref({ color: 'red' })
</script>

<template>
  <!-- 数组语法：多个样式对象合并 -->
  <p :style="[baseStyles, overrideStyles]">合并样式</p>
  <!-- 最终: color: red, fontSize: 16px（后面的覆盖前面的） -->
</template>
```

### 2.3 自动前缀

```vue
<template>
  <!-- Vue 自动添加浏览器前缀 -->
  <div :style="{ display: ['-webkit-box', '-ms-flexbox', 'flex'] }">
    <!-- Vue 会选择浏览器支持的第一个值 -->
  </div>
</template>
```

### 2.4 CSS 变量

```vue
<script setup>
import { ref } from 'vue'
const themeColor = ref('#42b883')
</script>

<template>
  <div :style="{ '--theme-color': themeColor }">
    <p style="color: var(--theme-color)">使用 CSS 变量</p>
  </div>
</template>
```

## 三、注意事项与常见陷阱

- `:style` 中的属性名可以使用 camelCase（`fontSize`）或 kebab-case（`font-size`）
- 样式值为数字时，Vue 会自动添加 `px` 后缀（部分属性除外如 `lineHeight`）
- 数组语法中后面的样式对象会覆盖前面的同名属性
- 使用 CSS 变量可以更灵活地控制样式
- `:style` 优先级高于外部样式表
