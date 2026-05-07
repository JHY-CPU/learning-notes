# useLocalStorage示例

## 一、概念说明

`useLocalStorage`实现响应式数据与localStorage的自动同步。修改ref值时自动写入localStorage，刷新页面时自动从localStorage读取。

```vue
<template>
  <div>
    <input v-model="username" placeholder="输入用户名" />
    <p>主题: {{ theme }}</p>
    <button @click="theme = theme === 'light' ? 'dark' : 'light'}">
      切换主题
    </button>
  </div>
</template>

<script setup>
import { useLocalStorage } from './composables/useLocalStorage'

const username = useLocalStorage('username', '游客')
const theme = useLocalStorage('theme', 'light')
</script>
```

## 二、具体用法

### 基础实现

```js
// composables/useLocalStorage.js
import { ref, watch } from 'vue'

export function useLocalStorage(key, defaultValue) {
  // 初始化时从localStorage读取
  const stored = localStorage.getItem(key)
  const data = ref(stored ? JSON.parse(stored) : defaultValue)

  // 监听变化，写入localStorage
  watch(data, (newVal) => {
    localStorage.setItem(key, JSON.stringify(newVal))
  }, { deep: true })

  return data
}
```

### 进阶实现 - 支持序列化配置

```js
export function useLocalStorage(key, defaultValue, options = {}) {
  const {
    serializer = JSON,
    onError = (e) => console.error(e)
  } = options

  const data = ref(defaultValue)

  try {
    const stored = localStorage.getItem(key)
    if (stored !== null) {
      data.value = serializer.parse(stored)
    }
  } catch (e) {
    onError(e)
  }

  watch(data, (val) => {
    try {
      localStorage.setItem(key, serializer.stringify(val))
    } catch (e) {
      onError(e)
    }
  }, { deep: true })

  return data
}
```

## 三、注意事项与常见陷阱

1. localStorage只能存储字符串，对象需要`JSON.stringify`
2. localStorage有容量限制（通常5MB），不要存储大数据
3. 监听`storage`事件可实现**跨标签页**同步
4. SSR环境下`localStorage`不可用，需做环境判断
5. 使用`watch`的`deep`选项确保嵌套对象变化也能被捕获
