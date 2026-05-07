# onRenderTriggered

## 一、概念说明
`onRenderTriggered` 是一个**调试钩子**，当响应式依赖的变化**触发了组件重新渲染**时调用。它可以帮助开发者追踪是哪个数据变化导致了重新渲染。

## 二、具体用法

### 2.1 基本用法
```vue
<script setup>
import { ref, onRenderTriggered } from 'vue'

const count = ref(0)

onRenderTriggered((event) => {
  console.log('渲染被触发:')
  console.log('  key:', event.key)          // 触发变化的 key
  console.log('  newValue:', event.newValue)
  console.log('  oldValue:', event.oldValue)
  console.log('  target:', event.target)
})
</script>

<template>
  <p>{{ count }}</p>
  <button @click="count++">+1</button>
</template>
<!-- 点击按钮时，控制台会输出触发信息 -->
```

### 2.2 追踪不必要的渲染
```vue
<script setup>
import { reactive, onRenderTriggered } from 'vue'

const state = reactive({ a: 1, b: 2, c: 3 })

onRenderTriggered(({ key, newValue, oldValue }) => {
  if (newValue === oldValue) {
    console.warn(`属性 ${key} 值未变但触发了渲染！`)
  }
})
</script>

<template>
  <p>{{ state.a }}</p>
  <!-- 修改 state.b 也会触发渲染，即使模板未使用 state.b -->
  <button @click="state.b++">改 b</button>
</template>
```

## 三、注意事项与常见陷阱
- **仅在开发环境**中可用
- 配合 `onRenderTracked` 可以分析完整的依赖链
- 如果未使用的数据变化触发了渲染，说明存在不必要的依赖
- 常用于性能优化：找出"不应该触发渲染却触发了"的原因
