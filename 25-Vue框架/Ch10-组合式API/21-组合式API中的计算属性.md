# 组合式API中的计算属性

## 一、概念说明

`computed`在组合式API中是一个函数，接收一个getter函数返回一个只读的响应式ref。也可以传入`{ get, set }`对象创建可写的计算属性。

```vue
<template>
  <div>
    <p>姓: {{ firstName }}</p>
    <p>名: {{ lastName }}</p>
    <p>全名: {{ fullName }}</p>
    <input v-model="fullName" placeholder="修改全名" />
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const firstName = ref('张')
const lastName = ref('三')

// 只读计算属性
const fullName = computed(() => `${firstName.value} ${lastName.value}`)

// 可写计算属性
const editableName = computed({
  get: () => `${firstName.value} ${lastName.value}`,
  set: (val) => {
    const [first, last] = val.split(' ')
    firstName.value = first
    lastName.value = last
  }
})
</script>
```

## 二、具体用法

### 基础用法

```js
import { ref, computed } from 'vue'

const count = ref(0)
const doubled = computed(() => count.value * 2)
const isEven = computed(() => count.value % 2 === 0)

console.log(doubled.value)  // 0
count.value = 5
console.log(doubled.value)  // 10
```

### 可写计算属性

```js
const firstName = ref('John')
const lastName = ref('Doe')

const fullName = computed({
  get() {
    return `${firstName.value} ${lastName.value}`
  },
  set(value) {
    [firstName.value, lastName.value] = value.split(' ')
  }
})

fullName.value = 'Jane Smith'
console.log(firstName.value)  // 'Jane'
```

## 三、注意事项与常见陷阱

1. `computed`返回的ref在JS中需要`.value`，模板中自动解包
2. 计算属性有**缓存**，依赖不变时不重新计算
3. 不要在getter中产生副作用（修改其他状态、异步操作）
4. 计算属性是只读的（除非定义setter）
5. 计算属性可以依赖其他计算属性，形成链式关系
