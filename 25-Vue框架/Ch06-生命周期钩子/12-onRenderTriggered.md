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

## 四、性能优化实例

```vue
<script setup>
import { reactive, onRenderTriggered, computed } from 'vue'

const state = reactive({
  list: Array.from({length: 100}, (_, i) => ({ id: i, text: `Item ${i}` })),
  filter: ''
})

// 追踪哪些属性频繁触发渲染
let triggerCount = {}
onRenderTriggered(({ key }) => {
  key = String(key);
  triggerCount[key] = (triggerCount[key] || 0) + 1;
  if (triggerCount[key] > 10) {
    console.warn(`属性 ${key} 已触发 ${triggerCount[key]} 次渲染，考虑优化`)
  }
})

// 优化方案：使用 computed 缓存
const filteredList = computed(() => {
  return state.list.filter(item => item.text.includes(state.filter))
})
</script>
```

## 五、结合 Vue DevTools

在开发环境中，`onRenderTriggered` 的信息与 Vue DevTools 的性能面板互补：

- DevTools 显示宏观的组件渲染瀑布图
- `onRenderTriggered` 提供细粒度的数据变化追踪
- 两者结合可以精确定位性能瓶颈

## 六、注意事项补充

- `onRenderTriggered` 只在开发环境中可用
- 在大型组件中使用时注意控制台输出量，可能会影响调试体验
- 如果组件从未触发 `onRenderTriggered`，说明它的响应式依赖没有变化
- 配合 `console.table` 可以更好地可视化触发记录
