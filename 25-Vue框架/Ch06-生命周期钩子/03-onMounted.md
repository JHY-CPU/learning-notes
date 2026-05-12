# onMounted

## 一、概念说明
`onMounted` 在组件**挂载到 DOM 之后**调用。此时组件的 DOM 节点已插入页面，可以安全地访问和操作 DOM 元素。这是最常用的生命周期钩子之一。

## 二、具体用法

### 2.1 访问 DOM 元素
```vue
<template>
  <div ref="containerRef">Hello</div>
</template>
<script setup>
import { ref, onMounted } from 'vue'

const containerRef = ref(null)

onMounted(() => {
  console.log(containerRef.value)  // <div>Hello</div>
  console.log(containerRef.value.offsetHeight)  // 元素高度
})
</script>
```

### 2.2 发起 API 请求
```vue
<script setup>
import { ref, onMounted } from 'vue'

const users = ref([])
const loading = ref(true)

onMounted(async () => {
  try {
    const res = await fetch('/api/users')
    users.value = await res.json()
  } finally {
    loading.value = false
  }
})
</script>
```

### 2.3 初始化第三方库
```vue
<template>
  <canvas ref="canvasRef"></canvas>
</template>
<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const canvasRef = ref(null)
let chart = null

onMounted(() => {
  chart = new Chart(canvasRef.value, { /* 配置 */ })
})

onUnmounted(() => {
  chart?.destroy()
})
</script>
```

### 2.4 全局事件监听
```vue
<script setup>
import { onMounted, onUnmounted } from 'vue'

function handleResize() {
  console.log('窗口大小:', window.innerWidth)
}

onMounted(() => window.addEventListener('resize', handleResize))
onUnmounted(() => window.removeEventListener('resize', handleResize))
</script>
```

## 三、常见用例

| 场景 | 说明 |
|------|------|
| DOM 操作 | 测量元素尺寸、聚焦输入框 |
| API 请求 | 组件挂载后获取数据 |
| 第三方库初始化 | 地图、图表、编辑器等 |
| 全局事件绑定 | resize、scroll、keyboard |

## 四、注意事项与常见陷阱

- 子组件的 `onMounted` 先于父组件执行（从内到外）
- `onMounted` 只在**客户端**调用，SSR 中不会执行
- 不要在 `onMounted` 中修改响应式数据导致无限更新
- ref 绑定的元素在 `onMounted` 之后才可用
- 如果需要等待所有子组件挂载完成，使用 `nextTick`
- `onMounted` 可以注册多个，按注册顺序执行
