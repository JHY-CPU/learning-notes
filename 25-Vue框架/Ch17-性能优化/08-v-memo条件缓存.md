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

## 四、性能对比实测

```vue
<script setup>
import { ref } from 'vue'

// 1000 项列表，只有 1 个选中项会变化
const list = ref(
  Array.from({ length: 1000 }, (_, i) => ({
    id: i,
    name: `项目 ${i}`,
    data: `这是第 ${i} 项的详细数据...`
  }))
)
const selectedId = ref(0)
</script>

<template>
  <!-- 没有 v-memo：每次选中都 diff 1000 个节点 -->
  <div v-for="item in list" :key="item.id">
    {{ item.name }} {{ item.id === selectedId ? '(选中)' : '' }}
  </div>

  <!-- 有 v-memo：只 diff 选中项，其余 999 项跳过 -->
  <div v-for="item in list" :key="item.id" v-memo="[item.id === selectedId]">
    {{ item.name }} {{ item.id === selectedId ? '(选中)' : '' }}
  </div>
</template>
```

## 五、v-memo 与其他优化手段对比

| 方式 | 灵活性 | 适用范围 | 性能开销 |
|------|--------|---------|---------|
| v-once | 完全静态 | 单个子树 | 最低 |
| v-memo | 条件缓存 | 列表/子树 | 低 |
| computed | 值缓存 | 单个值 | 低 |
| shallowRef | 浅层响应 | 大对象 | 低 |
| 虚拟滚动 | 按需渲染 | 超长列表 | 中 |

## 六、典型使用模式

```vue
<!-- 模式1：选中/收藏状态切换 -->
<div v-for="item in list" :key="item.id"
  v-memo="[item.id === activeId, item.isFavorite]">
  <!-- 只在选中或收藏状态变化时重新渲染 -->
</div>

<!-- 模式2：排序/过滤结果展示 -->
<div v-for="item in sortedList" :key="item.id"
  v-memo="[item.id, item.updatedAt]">
  <!-- 只在数据本身变化时重新渲染 -->
</div>

<!-- 模式3：整个组件子树的条件缓存 -->
<div v-memo="[userId, isAdmin]">
  <ExpensiveComponent :user-id="userId" :is-admin="isAdmin" />
</div>
```

## 三、注意事项与常见陷阱

- `v-memo` 依赖数组中的值必须是原始值或稳定引用
- 依赖数组为空 `v-memo="[]"` 等同于 `v-once`
- 仅在列表渲染中有显著性能收益时使用，不要滥用
- 依赖数组选择不当可能导致 UI 不更新（过时数据）
- `v-memo` 在列表中使用时必须配合 `:key`
- 复杂对象在依赖数组中比较的是引用，值变化需替换整个对象
