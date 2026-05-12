# customRef自定义引用

## 一、概念说明

`customRef`允许创建自定义的ref，显式控制依赖追踪和更新触发。常用于防抖、节流、验证等场景。

```vue
<template>
  <div>
    <input v-model="text" placeholder="输入文字（防抖）" />
    <p>实际值: {{ text }}</p>
  </div>
</template>

<script setup>
import { useDebouncedRef } from './composables/useDebouncedRef'

const text = useDebouncedRef('hello', 500)
</script>
```

## 二、具体用法

### 防抖Ref实现

```js
// composables/useDebouncedRef.js
import { customRef } from 'vue'

export function useDebouncedRef(value, delay = 300) {
  let timeout = null

  return customRef((track, trigger) => ({
    get() {
      track()  // 追踪依赖
      return value
    },
    set(newValue) {
      clearTimeout(timeout)
      timeout = setTimeout(() => {
        value = newValue
        trigger()  // 触发更新
      }, delay)
    }
  }))
}
```

### 只允许数字的Ref

```js
import { customRef } from 'vue'

export function useNumberRef(initialValue = 0) {
  let value = initialValue

  return customRef((track, trigger) => ({
    get() {
      track()
      return value
    },
    set(newValue) {
      const num = Number(newValue)
      if (!isNaN(num)) {
        value = num
        trigger()
      }
    }
  }))
}
```

### 带日志的Ref

```js
export function useLoggedRef(value, label = '') {
  return customRef((track, trigger) => ({
    get() {
      track()
      return value
    },
    set(newValue) {
      console.log(`[${label}] ${value} -> ${newValue}`)
      value = newValue
      trigger()
    }
  }))
}
```

## 三、注意事项与常见陷阱

1. `track()`必须在`get`中调用，否则依赖不会被收集
2. `trigger()`必须在值改变后调用，否则视图不会更新
3. 不在`get`中调用`track`会导致响应式失效
4. 自定义ref仍然可以在模板中直接使用，无需`.value`
5. 适用于需要控制更新时机、值验证等高级场景

## 四、实用 customRef 示例

### 4.1 节流 Ref
```js
import { customRef } from 'vue'

export function useThrottledRef(value, interval = 300) {
  let timer = null
  let lastTime = 0

  return customRef((track, trigger) => ({
    get() {
      track()
      return value
    },
    set(newValue) {
      const now = Date.now()
      if (now - lastTime >= interval) {
        value = newValue
        trigger()
        lastTime = now
      } else {
        clearTimeout(timer)
        timer = setTimeout(() => {
          value = newValue
          trigger()
          lastTime = Date.now()
        }, interval - (now - lastTime))
      }
    }
  }))
}
```

### 4.2 双向绑定验证 Ref
```js
import { customRef } from 'vue'

export function useValidatedRef(initialValue, validator) {
  let value = initialValue
  const error = customRef((track, trigger) => ({
    get() { track(); return value },
    set(newValue) {
      if (validator(newValue)) {
        value = newValue
        trigger()
      }
    }
  }))
  return error
}

// 使用
const email = useValidatedRef('', (v) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(v))
const age = useValidatedRef(0, (v) => v >= 0 && v <= 150)
```

### 4.3 带持久化的 Ref
```js
import { customRef } from 'vue'

export function usePersistedRef(key, defaultValue) {
  const stored = localStorage.getItem(key)
  let value = stored ? JSON.parse(stored) : defaultValue

  return customRef((track, trigger) => ({
    get() {
      track()
      return value
    },
    set(newValue) {
      value = newValue
      localStorage.setItem(key, JSON.stringify(newValue))
      trigger()
    }
  }))
}
```

## 五、customRef 与普通 ref 的性能对比

| 特性 | ref | customRef |
|------|-----|-----------|
| 创建开销 | 极低 | 略高 |
| 读取性能 | 最优 | 略低（额外函数调用）|
| 更新性能 | 最优 | 取决于实现 |
| 适用场景 | 通用 | 特殊需求 |
