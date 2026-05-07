# toRef 与 toRefs

## 一、概念说明

`toRefs` 将 reactive 对象的每个属性转换为独立的 ref，解构后保持响应式连接。`toRef` 为 reactive 对象的单个属性创建 ref。两者解决了解构 reactive 丢失响应式的问题。

```vue
<script setup>
import { reactive, toRefs, toRef } from 'vue'

const state = reactive({
  count: 0,
  name: 'Vue 3'
})

// toRefs: 转换所有属性
const { count, name } = toRefs(state)
count.value++  // 保持响应式，state.count 同步更新

// toRef: 转换单个属性
const countRef = toRef(state, 'count')
countRef.value++
</script>

<template>
  <p>{{ state.count }}</p>
  <button @click="count++">+1</button>
</template>
```

## 二、具体用法

### 2.1 toRefs 解构保持响应式

```vue
<script setup>
import { reactive, toRefs } from 'vue'

const state = reactive({
  firstName: '张',
  lastName: '三',
  age: 25
})

// 解构后每个属性都是独立的 ref
const { firstName, lastName, age } = toRefs(state)

// 修改 ref → 原始 state 同步更新
function changeName() {
  firstName.value = '李'
  lastName.value = '四'
}
</script>
```

### 2.2 toRef 单属性引用

```vue
<script setup>
import { reactive, toRef } from 'vue'

const state = reactive({ count: 0, name: 'Vue' })

// 只为 count 属性创建 ref
const countRef = toRef(state, 'count')

// 双向同步
countRef.value = 42
console.log(state.count) // 42
</script>
```

### 2.3 组合式函数中使用

```js
// composables/useMouse.js
import { reactive, toRefs } from 'vue'

export function useMouse() {
  const state = reactive({ x: 0, y: 0 })

  const update = (e) => {
    state.x = e.pageX
    state.y = e.pageY
  }

  onMounted(() => window.addEventListener('mousemove', update))
  onUnmounted(() => window.removeEventListener('mousemove', update))

  return toRefs(state) // 返回 ref，解构后保持响应式
}
```

## 三、注意事项与常见陷阱

- `toRefs` 只转换顶层属性，不会递归转换嵌套对象
- `toRef` 对不存在的属性返回 `undefined` 的 ref
- `toRefs` 返回的 ref 与原始属性双向同步
- 在组合式函数中推荐使用 `toRefs` 返回值
- 不要用 `toRefs` 转换非 reactive 对象
