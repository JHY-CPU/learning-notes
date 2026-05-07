# watchSyncEffect

## 一、概念说明

`watchSyncEffect` 是 `watchEffect` 的快捷方式，等价于 `watchEffect` 配置 `{ flush: 'sync' }`。它在响应式数据变化时**同步**执行回调，不等待 DOM 更新。这意味着每次变化都会立即触发，没有批处理。

```vue
<script setup>
import { ref, watchSyncEffect } from 'vue'

const count = ref(0)

// 同步执行：数据变化时立即触发
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

### 2.2 使用场景

```vue
<script setup>
import { ref, watchSyncEffect } from 'vue'

const input = ref('')

// 需要同步获取值的场景
watchSyncEffect(() => {
  // 每次输入都立即执行，无延迟
  validateSync(input.value)
})

function validateSync(value) {
  if (value.length > 10) {
    console.log('输入过长') // 立即同步输出
  }
}
</script>
```

## 三、注意事项与常见陷阱

- watchSyncEffect 没有批处理，同一 tick 内多次变化会多次执行
- 性能影响较大，只在确实需要同步执行时使用
- 在 watchSyncEffect 中修改响应式数据可能导致无限循环
- 大多数场景应该使用默认的 `watchEffect`（pre flush）
- 不要在 watchSyncEffect 中操作 DOM（DOM 尚未更新）
