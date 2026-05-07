# onBeforeMount

## 一、概念说明
`onBeforeMount` 在组件**挂载到 DOM 之前**调用。此时模板已编译完成（render 函数已生成），但 DOM 节点尚未创建并插入页面。

## 二、具体用法

### 2.1 基本用法
```vue
<script setup>
import { onBeforeMount, ref } from 'vue'

const status = ref('准备挂载')

onBeforeMount(() => {
  console.log('组件即将挂载')
  console.log('模板已编译，但 $el 还不存在')
  status.value = '即将挂载'
})
</script>
```

### 2.2 初始化非响应式数据
```vue
<script setup>
import { onBeforeMount } from 'vue'

// 在挂载前初始化第三方库配置
let chartInstance = null

onBeforeMount(() => {
  // 提前准备配置，但不操作 DOM
  chartInstance = { theme: 'dark', animate: true }
})
</script>
```

## 三、注意事项与常见陷阱
- 此阶段**无法访问 DOM 元素**，不要尝试操作 `$el` 或 `ref` 绑定的 DOM
- 服务端渲染（SSR）中此钩子**不会被调用**
- 大多数初始化逻辑应在 `onMounted` 中执行，而非 `onBeforeMount`
- 适合做不依赖 DOM 的准备工作
