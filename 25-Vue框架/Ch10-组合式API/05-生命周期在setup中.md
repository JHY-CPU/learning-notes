# 生命周期在setup中

## 一、概念说明

组合式API中的生命周期钩子是`on`前缀的函数，需要从Vue中导入。它们在`setup()`中注册，替代选项式API中的`mounted`、`created`等。

```vue
<template>
  <p>组件已运行: {{ elapsed }}秒</p>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const elapsed = ref(0)
let timer = null

onMounted(() => {
  console.log('组件挂载完成，DOM可用')
  timer = setInterval(() => elapsed.value++, 1000)
})

onUnmounted(() => {
  console.log('组件即将卸载，清理定时器')
  clearInterval(timer)
})
</script>
```

## 二、具体用法

### 选项式 vs 组合式生命周期对照

| 选项式API | 组合式API | 说明 |
|-----------|-----------|------|
| beforeCreate | setup() | setup即替代 |
| created | setup() | setup即替代 |
| beforeMount | onBeforeMount | 挂载前 |
| mounted | onMounted | 挂载完成 |
| beforeUpdate | onBeforeUpdate | 更新前 |
| updated | onUpdated | 更新完成 |
| beforeUnmount | onBeforeUnmount | 卸载前 |
| unmounted | onUnmounted | 卸载完成 |
| errorCaptured | onErrorCaptured | 捕获错误 |

### 常见使用场景

```vue
<script setup>
import { onMounted, onUpdated, onUnmounted } from 'vue'

onMounted(() => {
  // 发起网络请求
  // 操作DOM元素
  // 添加事件监听
})

onUpdated(() => {
  // DOM更新后执行
  // 访问更新后的DOM
})

onUnmounted(() => {
  // 清理定时器
  // 移除事件监听
  // 取消网络请求
})
</script>
```

## 三、注意事项与常见陷阱

1. 所有生命周期钩子**必须在setup()中同步注册**，不能在异步回调中使用
2. `onBeforeUnmount`和`onUnmounted`在SSR中不会被调用
3. `onMounted`中才能安全访问DOM元素
4. 可以注册**多个**同名钩子，按注册顺序执行
5. **没有`onCreated`**，因为`setup()`本身就相当于`created`
