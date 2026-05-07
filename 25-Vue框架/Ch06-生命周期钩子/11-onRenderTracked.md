# onRenderTracked

## 一、概念说明
`onRenderTracked` 是一个**调试钩子**，当组件的响应式依赖在渲染过程中**被追踪（track）**时触发。它可以帮助开发者了解组件渲染时依赖了哪些响应式数据。

## 二、具体用法

### 2.1 基本用法
```vue
<script setup>
import { ref, onRenderTracked } from 'vue'

const count = ref(0)
const name = ref('张三')

onRenderTracked((event) => {
  console.log('依赖被追踪:')
  console.log('  key:', event.key)       // 被追踪的属性名
  console.log('  target:', event.target) // 响应式对象
  console.log('  type:', event.type)     // 操作类型 (get/has/iterate)
})
</script>

<template>
  <p>{{ count }}</p>
  <p>{{ name }}</p>
</template>
<!-- 控制台会输出 count 和 name 被追踪的信息 -->
```

### 2.2 依赖分析
```vue
<script setup>
import { reactive, onRenderTracked } from 'vue'

const state = reactive({
  user: { name: '张三', age: 25 },
  settings: { theme: 'dark' }
})

onRenderTracked(({ key, target }) => {
  // 帮助识别哪些数据被实际使用
  console.log(`追踪依赖: ${key}`)
})
</script>
```

## 三、注意事项与常见陷阱
- **仅在开发环境**中可用，生产环境会被移除
- 主要用于调试和性能优化分析
- 不要在其中执行业务逻辑
- 与 `onRenderTriggered` 配合使用，完整追踪响应式依赖链
