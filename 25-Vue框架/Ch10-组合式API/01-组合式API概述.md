# 组合式API概述

## 一、概念说明

Vue 3 引入了**组合式API (Composition API)**，它是一组基于函数的API，允许我们更灵活地组织组件逻辑。相比选项式API，组合式API解决了逻辑关注点分散的问题。

```vue
<template>
  <div>
    <p>计数: {{ count }}</p>
    <p>双倍: {{ double }}</p>
    <button @click="increment">+1</button>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const count = ref(0)
const double = computed(() => count.value * 2)
const increment = () => count.value++
</script>
```

**为什么需要组合式API？**
- 选项式API中，相同逻辑的data、methods、computed分散在不同位置
- 组件变大后，逻辑关注点难以追踪
- 组合式API将同一逻辑关注点的代码集中在一起

## 二、具体用法

组合式API的核心是`setup()`函数（或`<script setup>`语法糖），在此作用域内可以使用：

- **响应式状态**：`ref()`、`reactive()`
- **计算属性**：`computed()`
- **生命周期钩子**：`onMounted()`、`onUnmounted()`等
- **侦听器**：`watch()`、`watchEffect()`

```vue
<script setup>
import { ref, reactive, computed, watch, onMounted } from 'vue'

// 响应式状态
const count = ref(0)
const user = reactive({ name: '张三', age: 20 })

// 计算属性
const info = computed(() => `${user.name} - ${user.age}岁`)

// 侦听器
watch(count, (newVal) => {
  console.log('计数变为:', newVal)
})

// 生命周期
onMounted(() => {
  console.log('组件已挂载')
})
</script>
```

## 三、注意事项与常见陷阱

1. **组合式API和选项式API可以共存**，但推荐选择一种风格保持一致
2. 在`<script setup>`中，顶层导入和变量自动暴露给模板
3. `ref`在JS中需要`.value`访问，在模板中自动解包
4. 组合式API更适合TypeScript类型推断
5. 逻辑复用通过**组合式函数 (Composables)** 实现，替代mixins
