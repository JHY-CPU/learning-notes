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
