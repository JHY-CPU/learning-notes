# Computed类型

## 一、概念说明

`computed()` 返回 `ComputedRef<T>` 类型，T 由计算函数的返回值自动推断。ComputedRef 是只读的 Ref，通过 `.value` 访问值。Vue 类型系统自动追踪 getter 中使用的响应式依赖，推断出正确的类型。

## 二、具体用法

### 自动类型推断

```ts
import { ref, computed } from 'vue'

const firstName = ref('张')
const lastName = ref('三')

// 自动推断类型为 ComputedRef<string>
const fullName = computed(() => `${firstName.value}${lastName.value}`)
// fullName.value 类型: string

const count = ref(10)
const price = ref(99)

// ComputedRef<number>
const total = computed(() => count.value * price.value)
// total.value = 990

// ComputedRef<boolean>
const isExpensive = computed(() => price.value > 50)
// isExpensive.value = true
```

### getter/setter 计算属性

```ts
import { ref, computed } from 'vue'

const firstName = ref('张')
const lastName = ref('三')

// 带 setter 的 computed
const fullName = computed({
  get() {
    return `${firstName.value} ${lastName.value}`
    // 返回类型: string
  },
  set(newValue: string) {
    const [first, ...rest] = newValue.split(' ')
    firstName.value = first
    lastName.value = rest.join(' ')
  }
})

fullName.value = '李 四'
// firstName.value = '李', lastName.value = '四'
```

### 复杂类型 Computed

```vue
<script setup lang="ts">
import { ref, computed } from 'vue'

interface User {
  id: number
  name: string
  role: 'admin' | 'editor' | 'viewer'
  active: boolean
}

const users = ref<User[]>([
  { id: 1, name: '张三', role: 'admin', active: true },
  { id: 2, name: '李四', role: 'editor', active: false },
  { id: 3, name: '王五', role: 'viewer', active: true }
])

// ComputedRef<User[]>
const activeUsers = computed(() =>
  users.value.filter(u => u.active)
)

// ComputedRef<string[]>
const userNames = computed(() =>
  users.value.map(u => u.name)
)

// ComputedRef<Record<string, User>>
const usersById = computed(() =>
  Object.fromEntries(users.value.map(u => [u.id, u]))
)
// usersById.value[1] 类型: User

// ComputedRef<{ admins: User[]; editors: User[]; viewers: User[] }>
const usersByRole = computed(() => ({
  admins: users.value.filter(u => u.role === 'admin'),
  editors: users.value.filter(u => u.role === 'editor'),
  viewers: users.value.filter(u => u.role === 'viewer')
}))
</script>

<template>
  <div>
    <p>活跃用户: {{ activeUsers.length }}</p>
    <p>所有姓名: {{ userNames.join(', ') }}</p>
    <p>管理员: {{ usersByRole.admins.map(u => u.name).join(', ') }}</p>
  </div>
</template>
```

## 三、注意事项与常见陷阱

1. **ComputedRef 是只读的**：不能直接赋值，需要通过 setter 或修改依赖数据
2. **getter 中不要有副作用**：computed 应是纯函数，不要在 getter 中修改状态
3. **避免在 computed 中执行异步操作**：computed 应同步返回值，异步用 watch
4. **computed 不会缓存对对象引用的修改**：返回新对象时每次都会触发更新
5. **循环依赖会导致无限递归**：computed A 依赖 B，B 依赖 A
