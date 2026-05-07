# 自定义指令与 TypeScript

## 一、概念说明
在 TypeScript 项目中，自定义指令需要正确的**类型定义**以获得类型检查和智能提示。

## 二、具体用法

### 2.1 指令类型定义
```ts
// directives/types.ts
import type { Directive, DirectiveBinding } from 'vue'

// 带类型的指令
export const vColor: Directive<HTMLElement, string> = {
  mounted(el, binding: DirectiveBinding<string>) {
    el.style.color = binding.value
  },
  updated(el, binding: DirectiveBinding<string>) {
    el.style.color = binding.value
  }
}
```

### 2.2 泛型指令
```ts
import type { Directive } from 'vue'

interface TooltipOptions {
  text: string
  position: 'top' | 'bottom' | 'left' | 'right'
}

export const vTooltip: Directive<HTMLElement, TooltipOptions> = {
  mounted(el, binding) {
    const { text, position = 'top' } = binding.value
    el.setAttribute('data-tooltip', text)
    el.setAttribute('data-position', position)
  }
}
```

### 2.3 在 script setup 中使用
```vue
<script setup lang="ts">
import type { Directive, DirectiveBinding } from 'vue'

// 局部指令，自动获得类型支持
const vFocus: Directive<HTMLInputElement> = {
  mounted(el: HTMLInputElement) {
    el.focus()
  }
}

const vColor: Directive<HTMLElement, string> = {
  mounted(el, binding) {
    el.style.color = binding.value
  }
}
</script>

<template>
  <input v-focus />
  <p v-color="'red'">红色文字</p>
</template>
```

### 2.4 全局类型声明
```ts
// shims-vue.d.ts
declare module 'vue' {
  export interface ComponentCustomProperties {
    vFocus: Directive
    vColor: Directive<HTMLElement, string>
  }
}
```

### 2.5 修饰符类型
```ts
export const vFormat: Directive<HTMLInputElement, string> = {
  mounted(el, binding) {
    const modifiers = binding.modifiers as Record<string, boolean>
    if (modifiers.uppercase) {
      el.addEventListener('input', () => {
        el.value = el.value.toUpperCase()
      })
    }
  }
}
```

## 三、注意事项与常见陷阱
- 使用 `Directive<Element, Value>` 泛型定义指令类型
- `DirectiveBinding` 泛型参数是 value 的类型
- 全局指令需要在类型声明文件中补充类型
- 在 `<script setup lang="ts">` 中定义的指令自动有类型推断
