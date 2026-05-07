# 模板引用 ref

## 一、概念说明

`ref` 用于在模板中获取 DOM 元素或子组件实例的引用。通过 `ref()` 声明一个与模板中 `ref` 属性同名的 ref，Vue 会在组件挂载后自动将 DOM 元素或组件实例赋值给该 ref。

```vue
<script setup>
import { ref, onMounted } from 'vue'

// 声明 ref（名称必须与模板中的 ref 属性匹配）
const inputRef = ref(null)

onMounted(() => {
  // 挂载后，inputRef.value 就是 DOM 元素
  inputRef.value.focus()
})
</script>

<template>
  <!-- ref 属性与 ref 变量名对应 -->
  <input ref="inputRef" placeholder="自动聚焦" />
</template>
```

## 二、具体用法

### 2.1 获取 DOM 元素

```vue
<script setup>
import { ref, onMounted } from 'vue'

const divRef = ref(null)
const canvasRef = ref(null)

onMounted(() => {
  // 操作 DOM
  divRef.value.style.backgroundColor = '#f0f0f0'

  // Canvas 操作
  const ctx = canvasRef.value.getContext('2d')
  ctx.fillStyle = '#42b883'
  ctx.fillRect(10, 10, 100, 100)
})
</script>

<template>
  <div ref="divRef">DOM 元素引用</div>
  <canvas ref="canvasRef" width="200" height="200"></canvas>
</template>
```

### 2.2 获取子组件实例

```vue
<script setup>
import { ref } from 'vue'
import ChildComponent from './ChildComponent.vue'

const childRef = ref(null)

function callChildMethod() {
  // 调用子组件暴露的方法
  childRef.value.someMethod()
}
</script>

<template>
  <ChildComponent ref="childRef" />
  <button @click="callChildMethod">调用子组件方法</button>
</template>
```

### 2.3 v-for 中的 ref

```vue
<script setup>
import { ref, onMounted } from 'vue'

const items = ref(['A', 'B', 'C'])
const itemRefs = ref([])

onMounted(() => {
  // itemRefs.value 是一个 DOM 元素数组
  console.log(itemRefs.value.map(el => el.textContent))
})
</script>

<template>
  <ul>
    <li v-for="item in items" :key="item" ref="itemRefs">
      {{ item }}
    </li>
  </ul>
</template>
```

### 2.4 函数式 ref

```vue
<script setup>
import { ref } from 'vue'

function setInputRef(el) {
  if (el) el.focus()
}
</script>

<template>
  <input :ref="setInputRef" />
</template>
```

## 三、注意事项与常见陷阱

- `ref` 在 `onMounted` 之前为 `null`，必须在挂载后访问
- `ref` 名称必须与模板中 `ref` 属性一致
- v-for 中使用 ref 会自动收集为数组
- 使用 `defineExpose` 暴露子组件的方法/属性给父组件
- 函数式 ref 在组件更新和卸载时会被调用（参数可能为 `null`）
