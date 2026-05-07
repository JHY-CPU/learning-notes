# watchPostEffect

## 一、概念说明

`watchPostEffect` 是 `watchEffect` 的快捷方式，等价于 `watchEffect` 配置 `{ flush: 'post' }`。它在 DOM 更新后执行回调，适合需要访问更新后 DOM 的场景。

```vue
<script setup>
import { ref, watchPostEffect } from 'vue'

const count = ref(0)
const boxRef = ref(null)

// 在 DOM 更新后执行
watchPostEffect(() => {
  if (boxRef.value) {
    console.log('元素宽度:', boxRef.value.offsetWidth)
  }
})
</script>

<template>
  <div ref="boxRef">{{ count }}</div>
  <button @click="count++">+1</button>
</template>
```

## 二、具体用法

### 2.1 访问更新后的 DOM

```vue
<script setup>
import { ref, watchPostEffect } from 'vue'

const list = ref([1, 2, 3])
const listRef = ref(null)

watchPostEffect(() => {
  // 确保在 DOM 更新后访问
  if (listRef.value) {
    const items = listRef.value.querySelectorAll('li')
    console.log(`列表项数: ${items.length}`)
  }
})
</script>

<template>
  <ul ref="listRef">
    <li v-for="item in list" :key="item">{{ item }}</li>
  </ul>
  <button @click="list.push(list.length + 1)">添加</button>
</template>
```

### 2.2 与 watchEffect 对比

```js
import { watchEffect, watchPostEffect, watchSyncEffect } from 'vue'

// 默认 flush: 'pre' - DOM 更新前
watchEffect(() => { /* ... */ })

// flush: 'post' - DOM 更新后
watchPostEffect(() => { /* ... */ })
// 等价于
watchEffect(() => { /* ... */ }, { flush: 'post' })

// flush: 'sync' - 同步
watchSyncEffect(() => { /* ... */ })
```

### 2.3 第三方库集成

```vue
<script setup>
import { ref, watchPostEffect, onMounted } from 'vue'
import Chart from 'chart.js'

const data = ref([10, 20, 30])
const canvasRef = ref(null)
let chart = null

watchPostEffect(() => {
  if (chart) {
    chart.data.datasets[0].data = data.value
    chart.update()
  }
})

onMounted(() => {
  chart = new Chart(canvasRef.value, {
    type: 'bar',
    data: { datasets: [{ data: data.value }] }
  })
})
</script>
```

## 三、注意事项与常见陷阱

- watchPostEffect 在 DOM 更新后执行，可以安全访问 DOM
- 不能在 watchPostEffect 中修改响应式数据（会导致无限循环）
- 组件首次渲染后也会执行一次
- 适合需要操作 DOM 或与第三方库集成的场景
- 与 `onUpdated` 生命周期不同，watchPostEffect 只追踪指定的依赖
