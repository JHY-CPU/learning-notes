# 类型化CSS

## 一、概念说明

在 Vue 3 + TypeScript 项目中，CSS 也能获得类型支持。CSS Modules 导出类型化的类名对象，`<style module>` 自动提供类型。样式变量和动态样式绑定也支持类型检查，避免拼写错误。

## 二、具体用法

### CSS Modules 类型

```vue
<script setup lang="ts">
// CSS Modules 自动类型化
import styles from './MyComponent.module.css'
// styles.container → string
// styles.active → string
// styles.highlight → string
</script>

<template>
  <div :class="styles.container">
    <p :class="{ [styles.active]: isActive }">内容</p>
    <span :class="styles.highlight">高亮</span>
  </div>
</template>

<style module>
.container {
  max-width: 1200px;
  margin: 0 auto;
}
.active {
  color: green;
  font-weight: bold;
}
.highlight {
  background: yellow;
  padding: 2px 4px;
}
</style>
```

### 动态样式类型

```vue
<script setup lang="ts">
// CSS 属性类型检查
const textColor = ref<string>('#333')
const fontSize = ref<number>(16)

// 对象形式的动态样式
const dynamicStyle = computed(() => ({
  color: textColor.value,
  fontSize: `${fontSize.value}px`,
  fontWeight: 'bold' as const
  // as const 确保字面量类型
}))

// 字符串形式
const inlineStyle = computed(() =>
  `color: ${textColor.value}; font-size: ${fontSize.value}px`
)

// 颜色联合类型
type ThemeColor = '#333' | '#42b883' | '#ff6b6b' | '#4ecdc4'
const theme = ref<ThemeColor>('#42b883')
</script>

<template>
  <div>
    <!-- 对象样式 -->
    <p :style="dynamicStyle">动态样式文本</p>

    <!-- 字符串样式 -->
    <p :style="inlineStyle">字符串样式文本</p>

    <!-- 主题色 -->
    <p :style="{ color: theme }">主题文本</p>

    <select v-model="theme">
      <option value="#333">深灰</option>
      <option value="#42b883">Vue绿</option>
      <option value="#ff6b6b">红色</option>
      <option value="#4ecdc4">青色</option>
    </select>
  </div>
</template>
```

### SCSS/Less 中的变量类型

```ts
// styles/variables.ts - 集中管理样式常量
export const colors = {
  primary: '#42b883',
  secondary: '#35495e',
  success: '#67c23a',
  warning: '#e6a23c',
  danger: '#f56c6c',
  info: '#909399'
} as const

export type ColorKey = keyof typeof colors
// ColorKey = 'primary' | 'secondary' | 'success' | 'warning' | 'danger' | 'info'

export const breakpoints = {
  sm: 640,
  md: 768,
  lg: 1024,
  xl: 1280
} as const
```

```vue
<script setup lang="ts">
import { colors, type ColorKey } from '@/styles/variables'

function getColor(name: ColorKey): string {
  return colors[name]
}

// getColor('primary') → '#42b883'
// getColor('invalid') → 类型错误
</script>
```

## 三、注意事项与常见陷阱

1. **CSS Modules 需要 .module.css 后缀**：普通 CSS 导入不会获得类型
2. **`$style` 是 `<style module>` 的默认注入名**：也可以自定义 module 名
3. **style 对象的属性名用 camelCase**：`font-size` 写为 `fontSize`
4. **类型定义文件可让 IDE 识别 CSS**：添加 `*.css.d.ts` 声明文件
5. **Tailwind CSS 不需要类型化**：通过 classnames 插件获得自动补全
