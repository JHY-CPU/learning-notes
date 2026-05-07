# watch 进阶选项

## 一、概念说明

`watch()` 接受第三个参数对象，提供 `deep`（深度侦听）、`immediate`（立即执行）、`once`（只触发一次）、`flush`（执行时机）等高级选项。

```vue
<script setup>
import { ref, reactive, watch } from 'vue'

const state = reactive({
  user: { name: '张三', profile: { age: 25 } }
})

// 深度侦听 + 立即执行
watch(
  () => state.user,
  (newVal) => {
    console.log('用户变化:', JSON.stringify(newVal))
  },
  { deep: true, immediate: true }
)
</script>
```

## 二、具体用法

### 2.1 deep 深度侦听

```vue
<script setup>
import { reactive, watch } from 'vue'

const state = reactive({
  nested: { count: 0 }
})

// 浅层侦听（默认）：只追踪顶层变化
watch(state, () => console.log('浅层触发'))

// 深度侦听：追踪所有嵌套变化
watch(state, () => console.log('深度触发'), { deep: true })

state.nested.count++ // 只有深度侦听会触发
</script>
```

### 2.2 immediate 立即执行

```vue
<script setup>
import { ref, watch } from 'vue'

const id = ref(1)

// immediate: 在创建时立即执行一次回调
watch(id, async (newId) => {
  const data = await fetch(`/api/users/${newId}`)
  // 处理数据...
}, { immediate: true })
</script>
```

### 2.3 once 只触发一次

```vue
<script setup>
import { ref, watch } from 'vue'

const ready = ref(false)

// once: 只触发一次后自动停止
watch(ready, () => {
  console.log('应用就绪！')
}, { once: true })
</script>
```

### 2.4 flush 执行时机

```vue
<script setup>
import { ref, watch } from 'vue'

const count = ref(0)

// flush: 'pre' (默认) - DOM 更新前
// flush: 'post' - DOM 更新后
// flush: 'sync' - 同步触发
watch(count, () => {
  console.log('DOM 更新后执行')
}, { flush: 'post' })
</script>
```

## 三、注意事项与常见陷阱

- `deep: true` 会追踪所有嵌套属性变化，性能开销较大
- 侦听 ref 时不需要 `deep: true`（ref 内部会自动深度侦听对象）
- `immediate` 会在 watch 创建时立即触发一次
- `once` 和 `immediate` 可以同时使用
- `flush: 'sync'` 可能导致不必要的重复渲染，谨慎使用
