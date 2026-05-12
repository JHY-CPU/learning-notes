# Store解构保持响应式

## 一、概念说明

直接解构Pinia store会丢失响应式。使用`storeToRefs()`将store属性转为ref，保持响应式连接。

```vue
<script setup>
import { useUserStore } from '@/stores/user'
import { storeToRefs } from 'pinia'

const userStore = useUserStore()

// ❌ 丢失响应式
const { name, age } = userStore

// ✅ 保持响应式
const { name, age, isLoggedIn } = storeToRefs(userStore)

// action不是ref，可直接解构
const { login, logout } = userStore
</script>
```

## 二、具体用法

### storeToRefs原理

```js
import { storeToRefs } from 'pinia'

const store = useCounterStore()

// storeToRefs 将每个state和getter转为ref
const refs = storeToRefs(store)
refs.count   // Ref<number> - 与store.count响应式同步
refs.doubled // Ref<number> - getter也转为computed ref
```

### 在模板中使用

```vue
<template>
  <!-- 模板中可直接用store.xxx（自动解包） -->
  <p>{{ userStore.name }}</p>
  <button @click="userStore.login">登录</button>
</template>

<script setup>
import { useUserStore } from '@/stores/user'
const userStore = useUserStore()
</script>
```

### 配合computed使用

```vue
<script setup>
import { computed } from 'vue'
import { useUserStore } from '@/stores/user'

const store = useUserStore()

// 不需要storeToRefs，直接用computed
const displayName = computed(() => `${store.name} (${store.age}岁)`)
</script>
```

## 四、解构最佳实践

```vue
<script setup>
import { useUserStore } from '@/stores/user'
import { storeToRefs } from 'pinia'
import { computed } from 'vue'

const userStore = useUserStore()

// 方式1：storeToRefs 解构（多个属性时推荐）
const { name, age, token, isLoggedIn } = storeToRefs(userStore)
const { login, logout, updateProfile } = userStore  // actions 直接解构

// 方式2：computed 包装（单个属性或派生值时推荐）
const greeting = computed(() => `你好，${userStore.name}`)

// 方式3：直接使用 store（模板中推荐）
// 模板中 userStore.name 自动解包
</script>

<template>
  <!-- 直接使用 store -->
  <p>{{ userStore.name }}</p>
  <button @click="userStore.login">登录</button>

  <!-- 使用解构的 ref -->
  <p>{{ name }}</p>
  <button @click="login">登录</button>

  <!-- 使用 computed -->
  <p>{{ greeting }}</p>
</template>
```

## 五、错误示例与正确示例

```js
const store = useCounterStore()

// ❌ 错误：丢失响应式
const { count, doubled } = store
// count 是数字，不会随 store 变化而更新

// ✅ 正确：保持响应式
const { count, doubled } = storeToRefs(store)
// count 是 Ref<number>，会随 store 变化

// ❌ 错误：action 不需要 storeToRefs
const { increment } = storeToRefs(store)
// action 不是响应式的，不需要转 ref

// ✅ 正确：action 直接解构
const { increment } = store
```

## 三、注意事项与常见陷阱

1. **永远不要**直接解构store的state和getter：`const { count } = store`
2. 使用`storeToRefs`解构state和getter
3. Action是普通函数，可直接解构：`const { increment } = store`
4. 在模板中使用`store.xxx`自动解包，无需额外处理
5. `storeToRefs`忽略actions和非响应式属性
6. `storeToRefs`返回的对象中，getter 被转为 `computed` ref
7. 如果只用一个属性，直接 `store.xxx` 比 `storeToRefs` 更简洁
