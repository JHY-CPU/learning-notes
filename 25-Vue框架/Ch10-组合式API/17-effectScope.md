# effectScope

## 一、概念说明

`effectScope`用于**批量管理**响应式副作用（computed、watch等）。当scope被停止时，其内的所有副作用同时停止，简化了清理工作。

```vue
<script setup>
import { effectScope, ref, watch, computed } from 'vue'

const scope = effectScope()

scope.run(() => {
  const count = ref(0)
  const doubled = computed(() => count.value * 2)

  watch(count, (val) => {
    console.log('count变化:', val)
  })

  // 这些副作用都在scope管理下
})

// 停止scope，清理所有副作用
function cleanup() {
  scope.stop()
}
</script>
```

## 二、具体用法

### 组合式函数中的scope管理

```js
// composables/useMouse.js
import { effectScope, ref, onMounted, onUnmounted } from 'vue'

export function useMouse() {
  const scope = effectScope()
  let x, y

  scope.run(() => {
    x = ref(0)
    y = ref(0)

    const handler = (e) => {
      x.value = e.clientX
      y.value = e.clientY
    }

    onMounted(() => window.addEventListener('mousemove', handler))
    onUnmounted(() => {
      window.removeEventListener('mousemove', handler)
      scope.stop()  // 停止scope内的所有副作用
    })
  })

  return { x, y }
}
```

### 嵌套scope

```js
const outerScope = effectScope()

outerScope.run(() => {
  const a = ref(1)

  const innerScope = effectScope()
  innerScope.run(() => {
    const b = ref(2)
    watch(b, () => console.log('b:', b.value))
  })

  // 停止内层scope
  innerScope.stop()

  // 外层scope不受影响
})
```

## 三、注意事项与常见陷阱

1. `effectScope`是Vue 3.1+的API，用于更精细的副作用管理
2. 停止scope后，其中的computed、watch全部失效且不可恢复
3. 主要用于库开发和复杂组合式函数
4. 在组件的setup中通常不需要手动创建scope（Vue自动管理）
5. 与`getCurrentScope()`配合可获取当前活跃的effect scope
