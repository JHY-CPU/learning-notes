# readonly 只读响应式

## 一、概念说明

`readonly()` 创建一个深层只读的响应式代理。任何修改操作（赋值、删除）都会在控制台发出警告，但不会生效。常用于防止子组件意外修改父组件传递的数据。

```vue
<script setup>
import { reactive, readonly } from 'vue'

const original = reactive({ count: 0 })
const copy = readonly(original)

// 修改原始数据
original.count++ // 正常工作

// 尝试修改只读代理
copy.count++ // 警告: Set operation on key "count" failed: target is readonly.
</script>
```

## 二、具体用法

### 2.1 shallowReadonly

```vue
<script setup>
import { shallowReadonly } from 'vue'

// 浅层只读：只有顶层属性只读
const state = shallowReadonly({
  count: 0,
  nested: { value: 1 }
})

state.count++ // 警告，不生效
state.nested.value++ // 可以修改（嵌套属性不是只读的）
</script>
```

### 2.2 使用场景

```vue
<script setup>
import { provide, readonly, ref } from 'vue'

const count = ref(0)

// provide 只读数据，防止子组件修改
provide('count', readonly(count))

// 子组件中无法修改
// inject('count').value++ // 警告
</script>
```

### 2.3 类型标注

```ts
import { readonly, type DeepReadonly } from 'vue'

const original = { nested: { count: 0 } }
const copy: DeepReadonly<typeof original> = readonly(original)
```

## 三、注意事项与常见陷阱

- readonly 是深层的，所有嵌套属性都是只读的
- 只读代理与原始对象是响应式关联的
- 修改只读数据不会报错，只是静默失败并发出警告
- `shallowReadonly` 只对顶层属性生效
- readonly 的类型是 `DeepReadonly<T>`，可与 TypeScript 配合
