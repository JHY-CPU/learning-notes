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

## 四、实际应用场景

### 4.1 大数据表格优化
```vue
<template>
  <table>
    <tr v-for="row in tableData" :key="row.id"
        v-memo="[row.id, row.status, selectedRows.has(row.id)]">
      <td>{{ row.name }}</td>
      <td>{{ row.email }}</td>
      <td :class="`status-${row.status}`">{{ row.status }}</td>
      <td>
        <input type="checkbox" :checked="selectedRows.has(row.id)" />
      </td>
    </tr>
  </table>
</template>
<script setup>
import { ref, reactive } from 'vue'

const tableData = ref(Array.from({ length: 10000 }, (_, i) => ({
  id: i,
  name: `User ${i}`,
  email: `user${i}@example.com`,
  status: 'active'
})))
const selectedRows = reactive(new Set())
</script>
```

### 4.2 虚拟列表中的缓存
```vue
<template>
  <div v-for="item in visibleItems" :key="item.id"
       v-memo="[item.id, item.isExpanded]">
    <div class="item-header">{{ item.title }}</div>
    <div v-if="item.isExpanded" class="item-body">
      {{ item.content }}
    </div>
  </div>
</template>
```

### 4.3 与计算属性配合
```vue
<script setup>
import { ref, computed } from 'vue'

const items = ref([])
const filter = ref('all')

// 先用 computed 过滤，再用 v-memo 缓存渲染
const filteredItems = computed(() => {
  if (filter.value === 'all') return items.value
  return items.value.filter(i => i.category === filter.value)
})
</script>

<template>
  <div v-for="item in filteredItems" :key="item.id"
       v-memo="[item.id, item.selected]">
    <span>{{ item.name }}</span>
    <span v-if="item.selected">✓</span>
  </div>
</template>
```

## 五、v-memo 的注意事项

```vue
<template>
  <!-- ⚠️ 依赖数组中引用不变时不会更新 -->
  <div v-for="item in items" :key="item.id" v-memo="[item]">
    <!-- 如果 item 是同一个引用，即使 item.name 变了也不会更新 -->
    {{ item.name }}
  </div>

  <!-- ✅ 用具体的值作为依赖 -->
  <div v-for="item in items" :key="item.id" v-memo="[item.id, item.name]">
    {{ item.name }}
  </div>
</template>
```

## 六、v-memo 与 React.memo 对比

| 特性 | Vue v-memo | React.memo |
|------|-----------|------------|
| 使用方式 | 模板指令 | HOC |
| 依赖声明 | 数组 | 浅比较/自定义 |
| 作用范围 | DOM 子树 | 组件 |
| 细粒度 | 更细（模板级别） | 组件级别 |
