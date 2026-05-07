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

## 三、注意事项与常见陷阱

- 列表中的每个元素**必须**有唯一的 `key`
- `<TransitionGroup>` 不支持 `mode` 属性（因为多个元素同时进出）
- CSS 的 `position: absolute` 元素不会参与移动过渡
