# 响应式 Provide

## 一、概念说明
默认情况下，`provide` 传递的值不是响应式的。要让后代组件能感知到数据的变化，需要传递 `ref`、`reactive` 对象或 `computed` 值。

## 二、具体用法

### 2.1 使用 ref 保持响应式
```vue
<!-- 祖先组件 -->
<script setup>
import { ref, provide } from 'vue'

const count = ref(0)
provide('count', count)
</script>

<template>
  <button @click="count++">计数: {{ count }}</button>
  <Child />
</template>
```

```vue
<!-- 后代组件 -->
<script setup>
import { inject } from 'vue'
const count = inject('count')
</script>

<template>
  <p>后代组件读取: {{ count }}</p>
</template>
```

### 2.2 使用 reactive 对象
```vue
<script setup>
import { reactive, provide } from 'vue'

const user = reactive({ name: '张三', age: 25 })
provide('user', user)
</script>
```

### 2.3 使用 computed 只读值
```vue
<script setup>
import { ref, computed, provide } from 'vue'

const count = ref(0)
const doubleCount = computed(() => count.value * 2)

provide('count', count)             // 后代可修改
provide('doubleCount', doubleCount) // 后代只读
</script>
```

### 2.4 使用 readonly 保护数据
```vue
<script setup>
import { ref, readonly, provide } from 'vue'

const state = ref({ count: 0 })
provide('state', readonly(state))  // 后代无法修改
</script>
```

### 2.5 封装 provide 函数
```ts
// composables/useTheme.ts
import { ref, provide, inject, type InjectionKey, type Ref } from 'vue'

const THEME_KEY: InjectionKey<Ref<string>> = Symbol('theme')

export function provideTheme() {
  const theme = ref('light')
  const toggle = () => {
    theme.value = theme.value === 'light' ? 'dark' : 'light'
  }
  provide(THEME_KEY, theme)
  return { theme, toggle }
}

export function useTheme() {
  const theme = inject(THEME_KEY)
  if (!theme) throw new Error('useTheme must be used within provideTheme')
  return theme
}
```

## 三、常见用例

| 场景 | 推荐方式 |
|------|---------|
| 主题切换 | ref + readonly 保护 |
| 用户信息 | reactive 对象 |
| 计数/统计数据 | computed 只读 |
| 操作方法 | provide 函数 |

## 四、注意事项与常见陷阱

- 传递 `ref` 时，后代组件会自动解包（不需要 `.value`）
- 使用 `readonly` 包裹可以防止后代意外修改数据
- `computed` 提供的值是只读的，适合暴露派生状态
- 响应式 provide 会让祖先和后代共享同一引用，注意数据所有权
- 后代修改 ref 数据会影响祖先组件，需谨慎设计
- 使用 TypeScript 的 InjectionKey 可以获得更好的类型安全
