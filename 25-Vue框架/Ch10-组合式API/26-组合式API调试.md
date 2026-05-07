# 组合式API调试

## 一、概念说明

Vue DevTools对组合式API提供了良好的调试支持。可以查看响应式依赖、组件状态、事件流等。`onRenderTriggered`和`onRenderTracked`是专用调试钩子。

```vue
<script setup>
import { ref, onRenderTracked, onRenderTriggered } from 'vue'

const count = ref(0)

// 追踪依赖收集
onRenderTracked((event) => {
  console.log('Tracked:', event.key, event.target)
})

// 追踪触发更新
onRenderTriggered((event) => {
  console.log('Triggered:', event.key, event.oldValue, event.newValue)
})
</script>
```

## 二、具体用法

### Vue DevTools调试

1. 安装Vue DevTools浏览器扩展
2. 打开DevTools的Vue面板
3. 选择组件查看：
   - `setup`状态中的ref/reactive值
   - computed的计算结果
   - 组件的props和events

### 自定义调试组合式函数

```js
// composables/useDebugRef.js
import { ref, watch } from 'vue'

export function useDebugRef(initialValue, label = 'debug') {
  const r = ref(initialValue)

  watch(r, (newVal, oldVal) => {
    console.log(`[${label}] ${oldVal} -> ${newVal}`)
    console.trace('变更来源')
  })

  return r
}
```

### 响应式调试API

```js
import { ref, computed, watch } from 'vue'

const count = ref(0)
const doubled = computed(() => count.value * 2)

// 通过console.table展示状态
function debugState() {
  console.table({
    count: count.value,
    doubled: doubled.value
  })
}
```

## 三、注意事项与常见陷阱

1. `onRenderTracked`仅在开发模式下生效
2. `onRenderTriggered`帮助定位不必要的重渲染
3. 使用`console.trace()`追踪状态变更的调用栈
4. Vue DevTools需要Vue 3.2+支持setup状态展示
5. 生产环境移除调试钩子，避免性能损耗
