# v-memo 缓存优化

## 一、概念说明
`v-memo` 是 Vue 3.2+ 引入的指令，用于**条件性地缓存模板子树**。它接收一个依赖数组，只有当数组中的值变化时才重新渲染，是性能优化的利器。

## 二、具体用法

### 2.1 基本用法
```vue
<template>
  <!-- 只有 item.id 变化时才重新渲染这个 li -->
  <li v-for="item in hugeList" :key="item.id" v-memo="[item.id]">
    <span>{{ item.name }}</span>
    <span>{{ item.price }}</span>
  </li>
</template>
<script setup>
import { ref } from 'vue'
const hugeList = ref(Array.from({ length: 10000 }, (_, i) => ({
  id: i, name: `Item ${i}`, price: Math.random() * 100
})))
</script>
```

### 2.2 多条件依赖
```vue
<template>
  <!-- 只有 isSelected 或 item.id 变化时才重新渲染 -->
  <div v-for="item in items" :key="item.id"
       v-memo="[item.id, isSelected(item.id)]">
    <span>{{ item.name }}</span>
    <span v-if="isSelected(item.id)">已选中</span>
  </div>
</template>
```

### 2.3 缓存整个子树
```vue
<template>
  <div v-memo="[shouldUpdate]">
    <!-- 这一整块内容只在 shouldUpdate 变化时重新渲染 -->
    <ExpensiveComponent />
    <p>{{ complexData }}</p>
  </div>
</template>
```

### 2.4 空数组（永不更新）
```vue
<template>
  <!-- 空数组 = 永不重新渲染，等价于 v-once -->
  <div v-memo="[]">
    静态内容
  </div>
</template>
```

## 三、注意事项与常见陷阱
- 只能在 Vue 3.2+ 使用
- 依赖数组中的值必须是响应式的
- 如果依赖数组中的引用不变，即使对象内部属性变化也不会更新
- 适合大数据列表优化，普通场景不需要使用
- 与 v-for 配合使用时，务必包含唯一标识（如 id）
