# 原始值的响应式 - RefImpl

## 一、概念说明

由于 Proxy 只能代理对象，Vue 3 使用 `RefImpl` 类包装原始值（string、number、boolean等）实现响应式。`RefImpl` 内部通过 `get value()` 和 `set value()` 的 getter/setter 实现依赖收集和触发更新。

```vue
<script setup>
import { ref } from 'vue'

// ref(0) 内部创建 RefImpl 对象
const count = ref(0)

// RefImpl 内部结构（简化）
// {
//   _value: 0,           // 内部存储的原始值
//   get value() { ... }, // 依赖收集
//   set value(v) { ... } // 触发更新
// }

console.log(count)       // RefImpl { _value: 0, ... }
console.log(count.value) // 0
</script>

<template>
  <!-- 模板中自动解包 RefImpl -->
  <p>{{ count }}</p>
  <button @click="count++">+1</button>
</template>
```

## 二、具体用法

### 2.1 为什么需要 .value

```js
import { ref } from 'vue'

const count = ref(0)

// .value 是访问 RefImpl 内部值的入口
console.log(count.value) // 0

// 没有 .value，你操作的是 RefImpl 对象本身
// count++ 不会工作，因为 count 是对象不是数字

// 必须通过 .value
count.value++
```

### 2.2 模板中自动解包

```vue
<script setup>
import { ref } from 'vue'
const count = ref(0)
const obj = ref({ nested: ref(1) })
</script>

<template>
  <!-- 自动解包: 不需要 .value -->
  <p>{{ count }}</p>
  <p>{{ obj.nested }}</p>
  <!-- 对象属性中的 ref 也会解包 -->
</template>
```

### 2.3 ref 包装对象

```js
import { ref, reactive } from 'vue'

// ref 包装对象时，内部调用 reactive()
const objRef = ref({ count: 0 })

// 这等价于
const objReactive = reactive({ count: 0 })

// objRef.value 是一个 reactive 代理
objRef.value.count++ // 响应式
```

## 三、注意事项与常见陷阱

- `.value` 在 JS 中必须使用，模板中自动解包
- 数组中的 ref 不会自动解包
- reactive 对象中的 ref 属性会自动解包
- `ref()` 对对象内部调用 `reactive()`，所以嵌套也是响应式的
- 不要用 `const { value } = ref(0)`，这会丢失响应式
