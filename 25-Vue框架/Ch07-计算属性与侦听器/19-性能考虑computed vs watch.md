# 性能考虑 computed vs watch

## 一、概念说明
`computed` 和 `watch` 在性能上有本质差异：computed 有缓存、惰性求值；watch 每次变化都执行回调。选择合适的工具可以显著影响应用性能。

## 二、性能对比

### 2.1 computed 的性能优势
```vue
<script setup>
import { ref, computed, watch } from 'vue'

const items = ref(/* 10000 个元素的大数组 */)

// ✅ computed: 缓存结果，只在 items 变化时计算一次
const total = computed(() =>
  items.value.reduce((sum, i) => sum + i.price, 0)
)

// ❌ watch: 每次变化都执行，且无法返回值
let cachedTotal = 0
watch(items, () => {
  cachedTotal = items.value.reduce((sum, i) => sum + i.price, 0)
})
</script>
```

### 2.2 watch 的适用场景
```vue
<script setup>
import { ref, watch } from 'vue'

const userId = ref(1)

// ✅ watch: 执行副作用（API 请求）
watch(userId, async (id) => {
  await fetchUserData(id)  // 需要执行副作用，而非返回值
})
</script>
```

### 2.3 避免在 watch 中做计算
```vue
<!-- ❌ 不好：用 watch 计算派生值 -->
<script setup>
import { ref, watch } from 'vue'
const price = ref(100)
const count = ref(3)
const total = ref(0)

watch([price, count], () => {
  total.value = price.value * count.value  // 应该用 computed
})
</script>

<!-- ✅ 好：用 computed -->
<script setup>
import { ref, computed } from 'vue'
const price = ref(100)
const count = ref(3)
const total = computed(() => price.value * count.value)
</script>
```

### 2.4 何时用哪个
```
返回一个值 → computed（有缓存、高效）
执行副作用 → watch / watchEffect
需要旧值  → watch
自动追踪  → watchEffect 或 computed
```

## 三、注意事项与常见陷阱
- 派生数据永远用 computed，不要用 watch 来"计算"
- watch 适合副作用：请求、DOM 操作、日志记录
- 大数组/对象的派生值必须用 computed，避免重复计算
- watchEffect 适合不需要 oldVal 且自动追踪依赖的副作用
