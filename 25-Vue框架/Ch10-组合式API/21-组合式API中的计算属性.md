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

## 四、在 Composable 中使用 computed

```js
import { ref, computed } from 'vue'

export function useShoppingCart() {
  const items = ref([])

  const totalPrice = computed(() =>
    items.value.reduce((sum, item) => sum + item.price * item.quantity, 0)
  )

  const discountedPrice = computed(() =>
    totalPrice.value > 200 ? totalPrice.value * 0.9 : totalPrice.value
  )

  const itemCount = computed(() =>
    items.value.reduce((sum, item) => sum + item.quantity, 0)
  )

  function addItem(item) {
    const existing = items.value.find(i => i.id === item.id)
    if (existing) {
      existing.quantity++
    } else {
      items.value.push({ ...item, quantity: 1 })
    }
  }

  return { items, totalPrice, discountedPrice, itemCount, addItem }
}
```

## 五、computed 的调试

```js
import { ref, computed } from 'vue'

const count = ref(0)

// 使用 getter 追踪
const doubled = computed(() => {
  const result = count.value * 2
  console.log(`[computed] ${count.value} * 2 = ${result}`)
  return result
})

// 使用 onTrack/onTrigger（开发工具）
const tracked = computed({
  get() { return count.value * 2 },
  onTrack(e) {
    console.log('依赖被追踪:', e)
  },
  onTrigger(e) {
    console.log('计算被触发:', e)
  }
})
```

## 六、computed 的注意事项

```js
// ❌ 不要在 computed 中产生副作用
const bad = computed(() => {
  saveToServer(count.value)  // 副作用！
  return count.value * 2
})

// ❌ 不要让 computed 依赖非响应式数据
let localCount = 0
const bad2 = computed(() => localCount * 2)  // 不会更新

// ✅ 正确的 computed
const good = computed(() => count.value * 2)
```
