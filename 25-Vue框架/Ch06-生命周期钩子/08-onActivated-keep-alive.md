# onActivated keep-alive

## 一、概念说明
`onActivated` 在被 `<keep-alive>` 缓存的组件**激活时**调用。当组件从缓存中恢复并重新显示时，此钩子会被触发，适合恢复组件状态或重新启动操作。

## 二、具体用法

### 2.1 基本用法
```vue
<!-- 父组件 -->
<template>
  <keep-alive>
    <ComponentA v-if="showA" />
    <ComponentB v-else />
  </keep-alive>
  <button @click="showA = !showA">切换</button>
</template>
```

```vue
<!-- ComponentA.vue -->
<script setup>
import { onActivated, onMounted, ref } from 'vue'

const status = ref('未激活')

onMounted(() => {
  console.log('首次挂载')
})

onActivated(() => {
  console.log('组件被激活（从缓存恢复）')
  status.value = '已激活'
})
</script>
```

### 2.2 恢复定时器/轮询
```vue
<script setup>
import { onActivated, onDeactivated } from 'vue'

let pollTimer = null

onActivated(() => {
  // 激活时恢复轮询
  pollTimer = setInterval(fetchData, 5000)
})

onDeactivated(() => {
  // 失活时暂停轮询
  clearInterval(pollTimer)
})
</script>
```

### 2.3 恢复滚动位置
```vue
<script setup>
import { ref, onActivated, onDeactivated } from 'vue'

const scrollPos = ref(0)
const containerRef = ref(null)

onDeactivated(() => {
  scrollPos.value = containerRef.value?.scrollTop || 0
})

onActivated(() => {
  containerRef.value?.scrollTo(0, scrollPos.value)
})
</script>
```

## 三、注意事项与常见陷阱
- 只有被 `<keep-alive>` 包裹的组件才会触发此钩子
- 首次挂载时 `onMounted` 和 `onActivated` 都会触发
- 之后每次从缓存恢复只触发 `onActivated`
- 此钩子在服务端渲染中不会被调用
