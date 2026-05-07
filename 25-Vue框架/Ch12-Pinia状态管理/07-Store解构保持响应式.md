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

## 三、注意事项与常见陷阱

1. **永远不要**直接解构store的state和getter：`const { count } = store`
2. 使用`storeToRefs`解构state和getter
3. Action是普通函数，可直接解构：`const { increment } = store`
4. 在模板中使用`store.xxx`自动解包，无需额外处理
5. `storeToRefs`忽略actions和非响应式属性
