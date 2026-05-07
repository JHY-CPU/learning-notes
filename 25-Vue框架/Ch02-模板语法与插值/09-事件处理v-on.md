# 事件处理 v-on

## 一、概念说明

`v-on` 指令用于监听 DOM 事件，简写为 `@`。可以绑定内联处理函数或方法调用。事件对象会自动传递给处理函数。

```vue
<script setup>
import { ref } from 'vue'

const count = ref(0)

function increment() {
  count.value++
}

function handleClick(event) {
  console.log('点击事件:', event.target)
}
</script>

<template>
  <!-- 完整写法 -->
  <button v-on:click="increment">+1</button>

  <!-- 简写（推荐） -->
  <button @click="increment">+1</button>

  <!-- 内联处理 -->
  <button @click="count++">+1</button>

  <!-- 传递事件对象 -->
  <button @click="handleClick($event)">点击我</button>
</template>
```

## 二、具体用法

### 2.1 事件传参

```vue
<script setup>
import { ref } from 'vue'

const items = ref([
  { id: 1, name: '项目 A' },
  { id: 2, name: '项目 B' }
])

function selectItem(item, event) {
  console.log('选中:', item.name)
  console.log('事件:', event.type)
}
</script>

<template>
  <div v-for="item in items" :key="item.id"
       @click="selectItem(item, $event)">
    {{ item.name }}
  </div>
</template>
```

### 2.2 多事件监听

```vue
<template>
  <!-- 同一元素监听多个事件 -->
  <input
    @focus="onFocus"
    @blur="onBlur"
    @input="onInput"
    @keydown.enter="onEnter"
  />
</template>
```

### 2.3 内联处理器中的方法

```vue
<script setup>
function warn(message, event) {
  if (event) {
    event.preventDefault()
  }
  alert(message)
}
</script>

<template>
  <!-- $event 是原生 DOM 事件对象 -->
  <button @click="warn('表单未完成', $event)">提交</button>
</template>
```

## 三、注意事项与常见陷阱

- 内联处理器中的方法调用会自动传入原生事件对象 `$event`
- 不要在模板中使用箭头函数（每次渲染创建新函数）
- 事件处理函数的 `this` 自动绑定为组件实例（选项式 API）
- `@click` 等价于 `v-on:click`，两者可互换
- 避免在模板中使用复杂的内联表达式
