# TransitionGroup 列表过渡

## 一、概念说明

`<TransitionGroup>` 用于为**列表**中的多个元素添加过渡效果。与 `<Transition>` 不同，它会渲染为一个真实的 DOM 元素（默认 `<span>`），且列表中每个元素都需要唯一的 `key`。

```vue
<script setup>
import { ref } from 'vue'
const items = ref([1, 2, 3, 4, 5])
let nextId = 6

function addItem() {
  items.value.splice(Math.floor(Math.random() * items.value.length), 0, nextId++)
}
function removeItem(index) {
  items.value.splice(index, 1)
}
</script>

<template>
  <button @click="addItem">添加</button>
  <TransitionGroup name="list" tag="ul">
    <li v-for="(item, index) in items" :key="item">
      {{ item }}
      <button @click="removeItem(index)">x</button>
    </li>
  </TransitionGroup>
</template>

<style>
.list-enter-active, .list-leave-active {
  transition: all 0.4s ease;
}
.list-enter-from { opacity: 0; transform: translateX(-30px); }
.list-leave-to { opacity: 0; transform: translateX(30px); }
</style>
```

## 二、具体用法

### 2.1 tag 属性

指定 `<TransitionGroup>` 渲染的容器标签，默认为 `<span>`。

```vue
<TransitionGroup tag="div" name="fade">
  <p v-for="item in list" :key="item.id">{{ item.text }}</p>
</TransitionGroup>
```

### 2.2 进入和离开

列表元素的**进入**和**离开**动画使用标准的 `enter-from` / `leave-to` 类名体系。

## 四、列表增删动画完整示例

```vue
<script setup>
import { ref } from 'vue'

const items = ref([
  { id: 1, text: '学习 Vue' },
  { id: 2, text: '学习 React' },
  { id: 3, text: '学习 TypeScript' }
])
let nextId = 4

const addItem = () => {
  items.value.push({ id: nextId++, text: `新任务 ${nextId - 1}` })
}

const removeItem = (id) => {
  const index = items.value.findIndex(item => item.id === id)
  if (index > -1) items.value.splice(index, 1)
}
</script>

<template>
  <button @click="addItem">添加任务</button>
  <TransitionGroup name="task" tag="ul" class="task-list">
    <li v-for="item in items" :key="item.id" class="task-item">
      <span>{{ item.text }}</span>
      <button @click="removeItem(item.id)">删除</button>
    </li>
  </TransitionGroup>
</template>

<style>
.task-list { list-style: none; padding: 0; }
.task-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px;
  margin: 5px 0;
  background: #f5f5f5;
  border-radius: 4px;
}
.task-enter-active, .task-leave-active {
  transition: all 0.4s ease;
}
.task-enter-from {
  opacity: 0;
  transform: translateX(-30px);
}
.task-leave-to {
  opacity: 0;
  transform: translateX(30px);
}
/* 移动过渡 */
.task-move {
  transition: transform 0.4s ease;
}
</style>
```

## 五、渲染标签与样式控制

```vue
<!-- 渲染为 <ul> -->
<TransitionGroup tag="ul" name="fade">
  <li v-for="item in items" :key="item.id">{{ item.text }}</li>
</TransitionGroup>

<!-- 渲染为 <div class="grid"> -->
<TransitionGroup tag="div" class="grid" name="cell">
  <div v-for="item in items" :key="item.id" class="cell">{{ item }}</div>
</TransitionGroup>
```

## 三、注意事项与常见陷阱

- 列表中的每个元素**必须**有唯一的 `key`
- `<TransitionGroup>` 不支持 `mode` 属性（因为多个元素同时进出）
- CSS 的 `position: absolute` 元素不会参与移动过渡
- 使用对象的 `id` 作为 `key`，而非数组 `index`
- 大量列表项（100+）考虑虚拟滚动 + TransitionGroup 的组合
