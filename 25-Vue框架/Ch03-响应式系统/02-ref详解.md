# ref 详解

## 一、概念说明

`ref()` 为任意值创建一个响应式引用。对于基本类型（string、number、boolean），ref 是创建响应式的唯一方式。通过 `.value` 属性访问和修改值。在模板中 ref 会自动解包，无需 `.value`。

```vue
<script setup>
import { ref } from 'vue'

const count = ref(0)
const name = ref('Vue 3')
const isActive = ref(true)

// 访问值
console.log(count.value) // 0

// 修改值
count.value++
name.value = 'Vue 3.4'

// 对象也可以用 ref
const user = ref({ name: '张三', age: 25 })
user.value.age = 26
</script>

<template>
  <!-- 模板中自动解包 -->
  <p>{{ count }}</p>
  <p>{{ name }}</p>
  <p>{{ isActive }}</p>
  <p>{{ user.name }} - {{ user.age }}</p>
  <button @click="count++">+1</button>
</template>
```

## 二、具体用法

### 2.1 ref 基本用法

```vue
<script setup>
import { ref } from 'vue'

const count = ref(0)
const double = () => count.value * 2

// ref 也可以包装对象
const obj = ref({ nested: { count: 0 } })
obj.value.nested.count++
</script>
```

### 2.2 ref 在模板中的解包

```vue
<script setup>
import { ref } from 'vue'

const count = ref(0)
const object = ref({ foo: 1 })
</script>

<template>
  <!-- 基本类型: 自动解包 -->
  <p>{{ count }}</p>     <!-- 不需要 .value -->

  <!-- 对象属性: 也自动解包 -->
  <p>{{ object.foo }}</p> <!-- 不需要 .value -->

  <!-- 但 object 本身不会解包 -->
  <!-- <p>{{ object }}</p> 显示 RefImpl 对象 -->
</template>
```

### 2.3 ref 的类型

```ts
import { ref, type Ref } from 'vue'

// 显式类型标注
const count = ref<number>(0)
const name = ref<string>('hello')

// Ref 类型
const myRef: Ref<number> = ref(42)
```

## 三、注意事项与常见陷阱

- `.value` 在 JS 中必须使用，模板中自动解包
- ref 作为 reactive 对象的属性时也会自动解包
- 数组中的 ref 不会自动解包
- `ref()` 返回的是 RefImpl 对象，不是原始值
- 修改 ref 时需要赋值 `.value`，直接赋值会丢失响应式
