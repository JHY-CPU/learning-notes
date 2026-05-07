# v-memo 条件缓存

## 一、概念说明

`v-memo` 是 Vue 3.2+ 新增的指令，接收一个依赖数组。只有当数组中的值发生变化时，才会重新渲染该子树。它比 `v-once` 更灵活，可以实现**条件性缓存**。

```vue
<script setup>
import { ref } from 'vue'
const list = ref([
  { id: 1, name: '项目A', selected: false },
  { id: 2, name: '项目B', selected: false },
  { id: 3, name: '项目C', selected: false },
])

const selectedId = ref(null)

function select(id) {
  selectedId.value = id
}
</script>

<template>
  <!-- 只有 item.id 和 selectedId 变化时才重新渲染对应项 -->
  <div
    v-for="item in list"
    :key="item.id"
    v-memo="[item.id === selectedId]"
    :class="{ active: item.id === selectedId }"
    @click="select(item.id)"
  >
    {{ item.name }} - {{ item.id === selectedId ? '选中' : '未选中' }}
  </div>
</template>
```

## 二、具体用法

### 2.1 大列表优化

```vue
<!-- 1000 项列表，只重新渲染选中项 -->
<li
  v-for="item in largeList"
  :key="item.id"
  v-memo="[item.id === activeId, item.isFavorite]"
>
  {{ item.name }}
</li>
```

### 2.2 依赖多个条件

```vue
<div v-memo="[userId, isAdmin, theme]">
  <!-- 只有这三个值变化时才重新渲染 -->
  <UserPanel :user-id="userId" :is-admin="isAdmin" :theme="theme" />
</div>
```

### 2.3 结合 v-if

```vue
<!-- v-memo 和 v-if 可以一起使用 -->
<div v-if="visible" v-memo="[data.version]">
  {{ expensiveRender(data) }}
</div>
```

## 三、注意事项与常见陷阱

- `v-memo` 依赖数组中的值必须是原始值或稳定引用
- 依赖数组为空 `v-memo="[]"` 等同于 `v-once`
- 仅在列表渲染中有显著性能收益时使用，不要滥用
- 依赖数组选择不当可能导致 UI 不更新（过时数据）
