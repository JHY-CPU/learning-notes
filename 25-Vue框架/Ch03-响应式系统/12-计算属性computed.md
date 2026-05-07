# 计算属性 computed

## 一、概念说明

`computed()` 创建一个响应式的计算值。计算属性会**缓存**结果，只有依赖的响应式数据变化时才重新计算。相比方法调用，computed 避免了不必要的重复计算。

```vue
<script setup>
import { ref, computed } from 'vue'

const firstName = ref('张')
const lastName = ref('三')

// 只读计算属性
const fullName = computed(() => {
  console.log('计算 fullName') // 只在依赖变化时执行
  return `${firstName.value} ${lastName.value}`
})
</script>

<template>
  <p>全名: {{ fullName }}</p>
  <!-- 多次使用 fullName 不会多次计算 -->
</template>
```

## 二、具体用法

### 2.1 只读计算属性

```vue
<script setup>
import { ref, computed } from 'vue'

const items = ref([
  { name: '苹果', price: 5, quantity: 3 },
  { name: '香蕉', price: 3, quantity: 5 }
])

const total = computed(() =>
  items.value.reduce((sum, item) => sum + item.price * item.quantity, 0)
)
</script>
```

### 2.2 可写计算属性

```vue
<script setup>
import { ref, computed } from 'vue'

const firstName = ref('张')
const lastName = ref('三')

const fullName = computed({
  get() {
    return `${firstName.value} ${lastName.value}`
  },
  set(value) {
    const [first, last] = value.split(' ')
    firstName.value = first
    lastName.value = last
  }
})

fullName.value = '李 四' // 触发 setter
</script>
```

### 2.3 computed vs methods

```vue
<script setup>
import { ref, computed } from 'vue'

const count = ref(0)

// computed: 有缓存，依赖不变返回缓存值
const doubled = computed(() => count.value * 2)

// method: 每次调用都重新计算
function getDoubled() {
  return count.value * 2
}
</script>
```

## 三、注意事项与常见陷阱

- computed 有缓存，性能优于在模板中调用方法
- 不要在 computed 中产生副作用（修改其他数据）
- computed 返回的是只读 Ref（除非使用 getter/setter）
- computed 的依赖必须是响应式的
- 避免在 computed 中进行异步操作
