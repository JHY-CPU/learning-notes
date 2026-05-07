# toRef与toRefs进阶

## 一、概念说明

`toRef`和`toRefs`用于将`reactive`对象的属性转换为独立的`ref`，**保持响应式连接**。它们解决了解构reactive对象后丢失响应式的问题。

```vue
<template>
  <p>姓名: {{ name }}</p>
  <p>年龄: {{ age }}</p>
  <button @click="age++">长一岁</button>
</template>

<script setup>
import { reactive, toRefs } from 'vue'

const user = reactive({ name: '张三', age: 20 })
const { name, age } = toRefs(user)
// name和age仍然是响应式的ref
</script>
```

## 二、具体用法

### toRef - 单个属性转换

```js
import { reactive, toRef } from 'vue'

const state = reactive({ count: 0, msg: 'hello' })

// 将单个属性转为ref
const countRef = toRef(state, 'count')
const msgRef = toRef(state, 'msg')

countRef.value++  // 同步修改 state.count
console.log(state.count) // 1
```

### toRefs - 所有属性转换

```js
import { reactive, toRefs } from 'vue'

const state = reactive({
  name: '李四',
  age: 25,
  address: { city: '北京' }
})

// 转换所有属性
const { name, age, address } = toRefs(state)

name.value = '王五'  // 同步修改 state.name
console.log(state.name) // '王五'
```

### 在组合式函数中使用

```js
export function useUser() {
  const user = reactive({
    name: '张三',
    age: 20,
    email: 'zhangsan@example.com'
  })

  // 使用toRefs返回，调用方可解构
  return toRefs(user)
}

// 使用
const { name, age } = useUser()  // 保持响应式
```

## 三、注意事项与常见陷阱

1. `toRef`对**不存在**的属性也能创建ref（值为`undefined`），不会报错
2. `toRefs`创建的ref与原对象**双向绑定**：修改ref会影响原对象，反之亦然
3. `toRefs`只处理**一层**属性，嵌套对象的属性仍需单独处理
4. 对`ref`使用`toRef`无意义，它已经是一个独立ref
5. **模板中解构props**时常用`toRefs`保持响应式：`const { title } = toRefs(props)`
