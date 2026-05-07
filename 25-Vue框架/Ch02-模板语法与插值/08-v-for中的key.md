# v-for 中的 key

## 一、概念说明

`key` 是 Vue 虚拟 DOM 算法用于识别节点身份的特殊属性。当数据变化时，Vue 通过 `key` 判断哪些元素需要更新、创建或销毁，从而高效地复用已有 DOM 节点。

没有 `key` 时，Vue 使用"就地更新"策略，可能导致状态错乱。

```vue
<script setup>
import { ref } from 'vue'

const items = ref([
  { id: 'a', text: '项目 A' },
  { id: 'b', text: '项目 B' },
  { id: 'c', text: '项目 C' }
])
</script>

<template>
  <!-- 正确: 使用唯一且稳定的 id 作为 key -->
  <div v-for="item in items" :key="item.id">
    {{ item.text }}
  </div>
</template>
```

## 二、具体用法

### 2.1 key 的选择策略

```vue
<template>
  <!-- 好: 使用数据中的唯一标识 -->
  <li v-for="user in users" :key="user.id">{{ user.name }}</li>

  <!-- 差: 使用数组索引（排序/增删时会出问题） -->
  <li v-for="(user, index) in users" :key="index">{{ user.name }}</li>

  <!-- 差: 不使用 key（Vue 会警告） -->
  <li v-for="user in users">{{ user.name }}</li>
</template>
```

### 2.2 key 不当导致的问题

```vue
<script setup>
import { ref } from 'vue'

// 假设有输入框列表
const items = ref([
  { id: 1, name: '项目 A' },
  { id: 2, name: '项目 B' }
])

// 如果用 index 作为 key:
// 在列表中间插入新项时，后续所有项的 key 都会变化
// 导致输入框的值错位
</script>

<template>
  <div v-for="(item, index) in items" :key="item.id">
    <input v-model="item.name" />
    <!-- 用 id: 输入框值正确绑定 -->
    <!-- 用 index: 插入时值会错位 -->
  </div>
</template>
```

### 2.3 key 与组件

```vue
<template>
  <!-- key 可以强制重新创建组件 -->
  <MyComponent :key="componentKey" />

  <!-- 通过改变 key 强制刷新组件 -->
  <button @click="componentKey++">刷新组件</button>
</template>
```

## 三、注意事项与常见陷阱

- `key` 应该是**唯一且稳定**的字符串或数字
- 避免使用 `index` 作为 key，除非列表不会排序或增删
- `key` 只在 `v-for` 的同一个作用域内需要唯一
- 用 `key` 强制组件重新渲染时，组件会被销毁和重建
- `v-if` 和 `v-for` 不要在同一元素上使用 key 冲突
