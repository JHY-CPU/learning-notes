# 清理副作用 onScopeDispose

## 一、概念说明
`onScopeDispose` 用于在**当前 effect scope 被销毁时**清理副作用。它与 `onUnmounted` 类似，但作用于 effect scope 层面，可以在组合式函数中使用而不依赖组件上下文。

## 二、具体用法

### 2.1 基本用法
```vue
<script setup>
import { onScopeDispose, effectScope, watch } from 'vue'

const scope = effectScope()
const someRef = ref(0)

scope.run(() => {
  watch(someRef, () => { /* ... */ })
})

onScopeDispose(() => {
  scope.stop()
  console.log('effect scope 已清理')
})
</script>
```

### 2.2 在 Composable 中使用
```js
// composables/useMouse.js
import { ref, onScopeDispose } from 'vue'

export function useMouse() {
  const x = ref(0)
  const y = ref(0)

  function update(e) {
    x.value = e.pageX
    y.value = e.pageY
  }

  window.addEventListener('mousemove', update)

  onScopeDispose(() => {
    window.removeEventListener('mousemove', update)
  })

  return { x, y }
}
```

### 2.3 与 onUnmounted 的区别
```js
// onScopeDispose: 可在 setup 外（composable 函数内）使用
// onUnmounted: 只能在 setup() 或 <script setup> 中使用

export function useTimer() {
  const timer = setInterval(() => {}, 1000)

  // 可以在 composable 中使用
  onScopeDispose(() => clearInterval(timer))

  // onUnmounted 在 setup 外不可用
}
```

### 2.4 effectScope 管理多个副作用
```js
import { effectScope, watch, computed, onScopeDispose } from 'vue'

export function useDashboard() {
  const scope = effectScope()

  scope.run(() => {
    const doubled = computed(() => count.value * 2)
    watch(count, () => updateChart())
    // 所有副作用都在 scope 内
  })

  // 一次性清理所有副作用
  onScopeDispose(() => scope.stop())

  return { doubled }
}
```

## 三、注意事项与常见陷阱
- `onScopeDispose` 不需要组件上下文，适合在组合式函数中使用
- 如果 effect scope 被手动停止，`onScopeDispose` 回调会立即执行
- 推荐在 composable 中使用 `onScopeDispose` 而非 `onUnmounted`
- 与 VueUse 库的清理机制一致

## 四、选择指南

| 场景 | 使用 |
| --- | --- |
| 在组件 `<script setup>` 中清理 | `onBeforeUnmount` |
| 在 composable 函数中清理 | `onScopeDispose` |
| 手动管理 effect scope | `scope.stop()` + `onScopeDispose` |
