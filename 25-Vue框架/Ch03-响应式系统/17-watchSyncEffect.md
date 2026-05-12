# watchSyncEffect

## 一、概念说明

`watchSyncEffect` 是 `watchEffect` 的快捷方式，等价于 `watchEffect` 配置 `{ flush: 'sync' }`。它在响应式数据变化时**同步**执行回调，不等待 DOM 更新。这意味着每次变化都会立即触发，没有批处理。

```vue
<script setup>
import { ref, watchSyncEffect } from 'vue'

const count = ref(0)

watchSyncEffect(() => {
  console.log('同步执行:', count.value)
})

count.value = 1 // 立即打印: 同步执行: 1
count.value = 2 // 立即打印: 同步执行: 2
</script>
```

## 二、具体用法

### 2.1 flush 选项对比

```js
import { watchEffect, watchPostEffect, watchSyncEffect } from 'vue'

// 'pre' (默认): DOM 更新前，有批处理
watchEffect(() => { /* ... */ })

// 'post': DOM 更新后，有批处理
watchPostEffect(() => { /* ... */ })

// 'sync': 同步执行，无批处理
watchSyncEffect(() => { /* ... */ })
```

### 2.2 同步验证

```vue
<script setup>
import { ref, watchSyncEffect } from 'vue'

const input = ref('')

watchSyncEffect(() => {
  if (input.value.length > 10) {
    console.log('输入过长') // 立即同步输出
  }
})
</script>
```

### 2.3 与 watch 对比

```js
import { ref, watch, watchSyncEffect } from 'vue'

const a = ref(0)
const b = ref(0)

// watch 默认是 pre flush（有批处理）
watch([a, b], ([newA, newB]) => {
  console.log(newA, newB) // 可能被合并
})

// watchSyncEffect 每次变化都立即执行
watchSyncEffect(() => {
  console.log(a.value, b.value) // 每次修改都立即打印
})

a.value = 1  // 立即打印 1 0
b.value = 2  // 立即打印 1 2
```

### 2.4 避免无限循环

```vue
<script setup>
import { ref, watchSyncEffect } from 'vue'

const count = ref(0)

// ⚠️ 危险：同步修改依赖会导致无限循环
watchSyncEffect(() => {
  console.log(count.value)
  // count.value++  // ❌ 永远不要这样做
})
</script>
```

## 三、常见用例

| 场景 | 推荐 |
|------|------|
| 需要立即响应的同步逻辑 | watchSyncEffect |
| 一般副作用 | watchEffect（默认 pre） |
| DOM 更新后的操作 | watchPostEffect |
| 需要访问旧值 | watch（不是 watchEffect） |

## 四、注意事项与常见陷阱

- watchSyncEffect 没有批处理，同一 tick 内多次变化会多次执行
- 性能影响较大，只在确实需要同步执行时使用
- 在 watchSyncEffect 中修改响应式数据可能导致无限循环
- 大多数场景应该使用默认的 `watchEffect`（pre flush）
- 不要在 watchSyncEffect 中操作 DOM（DOM 尚未更新）
- 频繁触发的场景（如输入框）不适合使用 watchSyncEffect
