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

## 四、调试工作流

```vue
<script setup>
import { ref, reactive, onRenderTracked, onRenderTriggered } from 'vue'

const count = ref(0)
const user = reactive({ name: 'Alice', age: 25 })

// 完整调试工作流：记录所有依赖
onRenderTracked(({ key, target, type }) => {
  console.log(`[TRACK] ${String(key)} - type: ${type}`)
})

onRenderTriggered(({ key, target, type, newValue, oldValue }) => {
  console.log(`[TRIGGER] ${String(key)} changed: ${oldValue} -> ${newValue}`)
  if (newValue === oldValue) {
    console.warn(`值未变但触发了渲染！`)
  }
})
</script>
```

## 五、排查不必要的依赖

如果 `onRenderTracked` 追踪到了模板中未使用的数据，说明组件可能因为模板中引用了不必要的变量而多余渲染。解决方法：

1. 将不必要的数据从模板中移除
2. 使用 `computed` 缓存派生数据
3. 使用 `shallowRef` 或 `markRaw` 减少深层追踪

## 六、注意事项补充

- `onRenderTracked` 在组件初始化时也会触发，不仅仅在更新时
- `event.target` 是响应式对象本身，`event.key` 是被追踪的属性名
- 生产环境中这些钩子会被 tree-shaking 移除，不影响性能
