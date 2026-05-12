# computed 基础

## 一、概念说明
`computed` 是一个**有缓存的计算属性**。它根据依赖的响应式数据自动计算值，只有依赖变化时才重新计算。相比 methods，computed 具有缓存优势。

## 二、具体用法

### 2.1 基本 getter
```vue
<script setup>
import { ref, computed } from 'vue'

const firstName = ref('张')
const lastName = ref('三')

// 只读计算属性
const fullName = computed(() => {
  return `${firstName.value} ${lastName.value}`
})
</script>

<template>
  <p>全名: {{ fullName }}</p>
</template>
```

### 2.2 自动依赖追踪
```vue
<script setup>
import { ref, computed } from 'vue'

const items = ref([
  { name: '苹果', price: 5, count: 3 },
  { name: '香蕉', price: 3, count: 5 }
])

// computed 自动追踪 items 的变化
const totalPrice = computed(() => {
  return items.value.reduce((sum, item) => sum + item.price * item.count, 0)
})
</script>
```

### 2.3 TypeScript 类型
```vue
<script setup>
import { ref, computed } from 'vue'

const count = ref(0)
// TypeScript 自动推断类型为 ComputedRef<number>
const doubled = computed(() => count.value * 2)
</script>
```

## 三、注意事项与常见陷阱
- computed 返回的是 `ComputedRef`，读取值需要 `.value`（模板中自动解包）
- 有**缓存**：依赖不变时，多次访问不会重新计算
- 不要在 computed 中执行副作用（如修改数据、发请求）
- computed 的依赖必须是响应式的，普通变量不会触发更新

## 四、computed vs methods 对比

```vue
<script setup>
import { ref, computed } from 'vue'

const items = ref([
  { name: '苹果', price: 5 },
  { name: '香蕉', price: 3 },
  { name: '橙子', price: 4 }
])

// computed：有缓存，依赖不变时不重新计算
const expensiveTotal = computed(() => {
  console.log('computed 被调用')  // 只在 items 变化时打印
  return items.value.reduce((sum, i) => sum + i.price, 0)
})

// methods：每次访问都执行
function calculateTotal() {
  console.log('method 被调用')  // 每次调用都打印
  return items.value.reduce((sum, i) => sum + i.price, 0)
}
</script>

<template>
  <!-- computed 缓存：多次访问只计算一次 -->
  <p>{{ expensiveTotal }} - {{ expensiveTotal }} - {{ expensiveTotal }}</p>
  <!-- console: "computed 被调用" 只打印 1 次 -->

  <!-- methods 无缓存：每次调用都执行 -->
  <p>{{ calculateTotal() }} - {{ calculateTotal() }} - {{ calculateTotal() }}</p>
  <!-- console: "method 被调用" 打印 3 次 -->
</template>
```

## 五、实际应用场景

### 5.1 搜索过滤
```vue
<script setup>
import { ref, computed } from 'vue'

const users = ref([
  { name: '张三', age: 25, city: '北京' },
  { name: '李四', age: 30, city: '上海' },
  { name: '王五', age: 28, city: '北京' }
])
const searchCity = ref('')

const filteredUsers = computed(() => {
  if (!searchCity.value) return users.value
  return users.value.filter(u => u.city === searchCity.value)
})
</script>
```

### 5.2 表单验证状态
```vue
<script setup>
import { ref, computed } from 'vue'

const email = ref('')
const password = ref('')
const confirmPassword = ref('')

const isEmailValid = computed(() => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.value))
const isPasswordValid = computed(() => password.value.length >= 8)
const doPasswordsMatch = computed(() => password.value === confirmPassword.value)
const isFormValid = computed(() => isEmailValid.value && isPasswordValid.value && doPasswordsMatch.value)
</script>

<template>
  <input v-model="email" :class="{ invalid: email && !isEmailValid }" />
  <input v-model="password" type="password" :class="{ invalid: password && !isPasswordValid }" />
  <input v-model="confirmPassword" type="password" :class="{ invalid: confirmPassword && !doPasswordsMatch }" />
  <button :disabled="!isFormValid">提交</button>
</template>
```

### 5.3 列表统计
```vue
<script setup>
import { ref, computed } from 'vue'

const orders = ref([
  { status: 'completed', amount: 100 },
  { status: 'pending', amount: 200 },
  { status: 'completed', amount: 150 },
  { status: 'cancelled', amount: 50 }
])

const completedOrders = computed(() =>
  orders.value.filter(o => o.status === 'completed')
)
const totalRevenue = computed(() =>
  completedOrders.value.reduce((sum, o) => sum + o.amount, 0)
)
const completionRate = computed(() =>
  orders.value.length ? (completedOrders.value.length / orders.value.length * 100).toFixed(1) + '%' : '0%'
)
</script>
```

## 六、调试 computed

```js
import { ref, computed } from 'vue'

const count = ref(1)

// 使用 getter 追踪计算过程
const doubled = computed(() => {
  const result = count.value * 2
  console.log(`[computed doubled] ${count.value} * 2 = ${result}`)
  return result
})

// 使用 onTrack/onTrigger 调试（仅开发模式）
const debugged = computed({
  get() { return count.value * 2 },
  onTrack(e) { console.log('tracked:', e) },
  onTrigger(e) { console.log('triggered:', e) }
})
```
