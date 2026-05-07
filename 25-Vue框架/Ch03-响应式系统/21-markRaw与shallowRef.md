# markRaw 与 shallowRef

## 一、概念说明

`markRaw()` 标记一个对象，使其永远不会被响应式系统转换。`shallowRef()` 只追踪 `.value` 的替换。两者结合使用可以高效处理大型不可变数据或第三方库实例。

```vue
<script setup>
import { markRaw, shallowRef, reactive } from 'vue'

// markRaw: 标记对象不转为响应式
const rawObj = markRaw({ count: 0 })
const state = reactive({ data: rawObj })
// state.data 永远不会是响应式代理

// shallowRef: 只追踪 .value 的替换
const chartInstance = shallowRef(null)
// chartInstance.value.xxx 变化不会触发更新
</script>
```

## 二、具体用法

### 2.1 markRaw 使用

```vue
<script setup>
import { reactive, markRaw } from 'vue'

// 第三方库实例不需要响应式
import { Editor } from 'some-editor-lib'

const state = reactive({
  content: '',                    // 需要响应式
  editor: markRaw(new Editor())  // 不需要响应式
})

// 修改 editor 实例不会触发不必要的更新
state.content = 'new content' // 触发更新
state.editor.focus()          // 不触发更新
</script>
```

### 2.2 shallowRef 使用

```vue
<script setup>
import { shallowRef } from 'vue'

const heavyData = shallowRef({
  items: new Array(10000).fill({ name: 'item' })
})

// 整体替换触发更新
function replaceData(newData) {
  heavyData.value = newData
}

// 修改内部属性不触发更新
heavyData.value.items[0].name = 'changed' // 不响应
</script>
```

### 2.3 两者对比

| 特性 | markRaw | shallowRef |
|------|---------|------------|
| 作用 | 标记对象不转为响应式 | 只追踪 .value 替换 |
| 层级 | 对象级别 | ref 级别 |
| 嵌套 | 标记后深层也不转 | 只看 .value |
| 典型场景 | 第三方库实例 | 大型不可变数据 |

## 三、注意事项与常见陷阱

- `markRaw` 是永久标记，不能撤销
- `markRaw` 标记的对象在任何响应式上下文中都不会被转换
- `shallowRef` 的 `.value` 可以是任何类型
- 不要对需要深层响应的数据使用 `markRaw` 或 `shallowRef`
- `markRaw` 只标记当前对象，不标记嵌套对象
