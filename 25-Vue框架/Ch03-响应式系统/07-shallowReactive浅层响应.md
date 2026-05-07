# shallowReactive 浅层响应

## 一、概念说明

`shallowReactive()` 创建一个浅层响应式代理。只有对象的顶层属性是响应式的，嵌套对象的属性变化不会触发更新。适用于只需要追踪顶层属性变化的场景。

```vue
<script setup>
import { shallowReactive } from 'vue'

const state = shallowReactive({
  count: 0,         // 顶层属性：响应式
  nested: {
    deep: 'value'   // 嵌套属性：非响应式
  }
})

state.count++ // 触发更新
state.nested.deep = 'new' // 不触发更新
</script>
```

## 二、具体用法

### 2.1 基本使用

```vue
<script setup>
import { shallowReactive } from 'vue'

const form = shallowReactive({
  username: '',   // 响应式
  email: '',      // 响应式
  settings: {     // 对象本身是响应式
    theme: 'dark'  // 但此属性不是响应式的
  }
})
</script>
```

### 2.2 与 reactive 对比

```js
import { reactive, shallowReactive } from 'vue'

const deep = reactive({ nested: { count: 0 } })
const shallow = shallowReactive({ nested: { count: 0 } })

deep.nested.count++   // 触发更新
shallow.nested.count++ // 不触发更新
```

### 2.3 使用场景

```vue
<script setup>
import { shallowReactive } from 'vue'

// 场景: 大型第三方库实例
const editor = shallowReactive({
  content: '',        // 需要响应式
  instance: null,     // 第三方库实例，不需要深度响应式
  isReady: false      // 需要响应式
})
</script>
```

## 三、注意事项与常见陷阱

- shallowReactive 只追踪顶层属性的增删改
- 嵌套对象的属性变化不会触发视图更新
- 适用于包含大型第三方库实例的对象
- 对顶层属性赋新对象是响应式的
- 与 `shallowRef` 的区别：shallowRef 以值为单位，shallowReactive 以顶层属性为单位
