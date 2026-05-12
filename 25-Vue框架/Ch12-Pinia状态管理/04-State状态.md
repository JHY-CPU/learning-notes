# State状态

## 一、概念说明

State是Store的核心，定义Store中存储的数据。在Pinia中可以直接读取和修改state，它是响应式的。

```js
import { defineStore } from 'pinia'

export const useAppStore = defineStore('app', {
  state: () => ({
    theme: 'light',
    language: 'zh-CN',
    sidebar: { collapsed: false, width: 240 },
    notifications: []
  })
})
```

## 二、具体用法

### 访问State

```vue
<script setup>
import { useAppStore } from '@/stores/app'

const store = useAppStore()

// 直接访问
console.log(store.theme)

// 解构（丢失响应式！）
const { theme } = store  // ❌ 不是响应式

// 使用storeToRefs解构（保持响应式）
import { storeToRefs } from 'pinia'
const { theme, language } = storeToRefs(store)  // ✅
</script>
```

### 修改State

```js
const store = useAppStore()

// 方式1：直接修改
store.theme = 'dark'

// 方式2：批量修改
store.$patch({
  theme: 'dark',
  language: 'en-US'
})

// 方式3：$patch函数形式（适合数组操作）
store.$patch((state) => {
  state.notifications.push({ id: 1, text: '新消息' })
  state.sidebar.collapsed = true
})

// 方式4：通过action（推荐）
store.toggleTheme()
```

### 重置State

```js
store.$reset()  // 恢复到初始状态
```

## 三、注意事项与常见陷阱

1. `state`必须是函数，保证每个实例独立
2. 直接解构`store`会丢失响应式，使用`storeToRefs`
3. `$patch`批量修改比逐个修改性能更好
4. `$reset()`只在选项式Store中可用，Setup式需自行实现
5. 添加新属性时使用`$patch`或确保属性在state中已定义

## 四、State 的高级操作

### 4.1 $patch 的性能优势
```js
// ❌ 逐个修改：每次修改都触发一次响应式更新
store.name = '新名字'
store.age = 25
store.email = 'new@example.com'

// ✅ $patch 批量修改：只触发一次响应式更新
store.$patch({
  name: '新名字',
  age: 25,
  email: 'new@example.com'
})
```

### 4.2 使用 $state 替换整个状态
```js
// 替换整个 state（用于从服务端恢复状态）
store.$state = {
  theme: 'dark',
  language: 'en',
  sidebar: { collapsed: true, width: 200 },
  notifications: []
}
```

### 4.3 订阅 state 变化
```js
store.$subscribe((mutation, state) => {
  // mutation.type: 'direct' | 'patch object' | 'patch function'
  // mutation.storeId: store 的 ID

  // 自动保存到 localStorage
  localStorage.setItem('app-state', JSON.stringify(state))
}, { detached: true })  // detached: 组件卸载后继续监听
```

### 4.4 State 中的非序列化数据
```js
export const useMapStore = defineStore('map', {
  state: () => ({
    // Map、Set 等复杂类型
    markers: new Map(),
    selectedIds: new Set(),
    // 正则、日期等
    filter: { pattern: /^test/, date: new Date() }
  })
})
```
