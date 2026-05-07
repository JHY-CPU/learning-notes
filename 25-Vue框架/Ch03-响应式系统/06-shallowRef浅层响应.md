# shallowRef 浅层响应

## 一、概念说明

`shallowRef()` 创建一个浅层响应式的 ref。只有 `.value` 的替换是响应式的，`.value` 内部的属性变化不会触发更新。适用于大型不可变数据或外部状态管理。

```vue
<script setup>
import { shallowRef, triggerRef } from 'vue'

const state = shallowRef({
  nested: { count: 0 }
})

// 替换 .value → 触发更新
state.value = { nested: { count: 1 } } // 响应式

// 修改内部属性 → 不触发更新
state.value.nested.count++ // 不响应！
</script>
```

## 二、具体用法

### 2.1 基本使用

```vue
<script setup>
import { shallowRef } from 'vue'

// 大型数据对象，只关心整体替换
const largeData = shallowRef({
  items: new Array(10000).fill({ id: 1, data: '...' })
})

// 只有整体替换才触发更新
function updateData(newData) {
  largeData.value = newData // 响应式
}
</script>
```

### 2.2 强制触发更新

```vue
<script setup>
import { shallowRef, triggerRef } from 'vue'

const shallow = shallowRef({ count: 0 })

function increment() {
  shallow.value.count++
  triggerRef(shallow) // 手动触发更新
}
</script>
```

### 2.3 与 ref 对比

```js
import { ref, shallowRef } from 'vue'

const deep = ref({ nested: { count: 0 } })
const shallow = shallowRef({ nested: { count: 0 } })

deep.value.nested.count++    // 触发更新
shallow.value.nested.count++ // 不触发更新
```

## 三、注意事项与常见陷阱

- shallowRef 只追踪 `.value` 的替换，不追踪内部属性
- 适合处理大型不可变数据或与外部状态管理库集成
- `triggerRef()` 可以手动触发 shallowRef 的更新
- 与 `shallowReactive` 不同，shallowRef 以整个值为单位
- Vue 官方的 `ref()` 对大型对象也建议用 shallowRef 优化
