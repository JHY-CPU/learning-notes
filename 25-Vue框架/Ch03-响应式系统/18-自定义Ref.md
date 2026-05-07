# 自定义 Ref - customRef

## 一、概念说明

`customRef()` 允许自定义 ref 的依赖追踪和触发更新逻辑。最典型的应用是实现**防抖 ref**（debounced ref），在用户停止输入一段时间后才更新数据。

```vue
<script setup>
import { customRef } from 'vue'

// 防抖 ref 实现
function useDebouncedRef(value, delay = 500) {
  let timeout
  return customRef((track, trigger) => ({
    get() {
      track() // 追踪依赖
      return value
    },
    set(newValue) {
      clearTimeout(timeout)
      timeout = setTimeout(() => {
        value = newValue
        trigger() // 触发更新
      }, delay)
    }
  }))
}

const text = useDebouncedRef('')
</script>

<template>
  <input v-model="text" placeholder="输入内容（防抖 500ms）" />
  <p>值: {{ text }}</p>
</template>
```

## 二、具体用法

### 2.1 customRef 基本结构

```js
import { customRef } from 'vue'

const myRef = customRef((track, trigger) => {
  let value = 'initial'

  return {
    get() {
      track() // 调用 track 告诉 Vue 这个数据被读取了
      return value
    },
    set(newValue) {
      value = newValue
      trigger() // 调用 trigger 告诉 Vue 数据变化了
    }
  }
})
```

### 2.2 验证 ref

```vue
<script setup>
import { customRef } from 'vue'

function useValidatedRef(initial, validator) {
  return customRef((track, trigger) => {
    let value = initial

    return {
      get() {
        track()
        return value
      },
      set(newValue) {
        if (validator(newValue)) {
          value = newValue
          trigger()
        }
        // 验证失败不更新
      }
    }
  })
}

const age = useValidatedRef(0, (val) => val >= 0 && val <= 150)
</script>

<template>
  <input v-model.number="age" type="number" />
  <p>年龄: {{ age }}</p>
</template>
```

### 2.3 只读 customRef

```js
const readOnlyRef = customRef((track, trigger) => ({
  get() {
    track()
    return someValue
  },
  set() {
    console.warn('此 ref 是只读的')
  }
}))
```

## 三、注意事项与常见陷阱

- `track()` 必须在 `get` 中调用，否则不会收集依赖
- `trigger()` 必须在 `set` 中调用，否则不会触发更新
- 忘记调用 `track()` 会导致数据不响应
- 忘记调用 `trigger()` 会导致视图不更新
- customRef 的值没有类型限制
