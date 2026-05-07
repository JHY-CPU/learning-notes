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
