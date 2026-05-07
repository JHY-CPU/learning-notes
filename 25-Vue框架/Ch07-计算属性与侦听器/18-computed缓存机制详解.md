# computed 缓存机制详解

## 一、概念说明
`computed` 的核心特性是**缓存**。它通过脏检查（dirty checking）机制实现惰性求值：只有依赖变化时标记为"脏"，下次访问时才重新计算。依赖未变时直接返回缓存值。

## 二、缓存机制原理

### 2.1 惰性求值
```vue
<script setup>
import { ref, computed } from 'vue'

const items = ref([1, 2, 3, 4, 5])

const total = computed(() => {
  console.log('执行计算')  // 只有在 items 变化后首次访问才执行
  return items.value.reduce((a, b) => a + b, 0)
})

// 多次访问，只计算一次
console.log(total.value)  // 执行计算，输出 15
console.log(total.value)  // 直接返回缓存，不输出"执行计算"
console.log(total.value)  // 直接返回缓存

items.value.push(6)  // 标记为脏
console.log(total.value)  // 执行计算，输出 21
</script>
```

### 2.2 依赖追踪过程
```
1. computed 首次执行 getter → 追踪访问的响应式属性
2. 依赖变化 → computed 标记为 dirty
3. 下次访问 computed → 重新执行 getter → 更新缓存 → 标记为 clean
4. 依赖未变 → 直接返回缓存值
```

### 2.3 与 methods 的对比
```vue
<script setup>
import { ref, computed } from 'vue'

const count = ref(0)

// computed: 缓存，count 不变时不重新计算
const doubled = computed(() => {
  console.log('computed 执行')
  return count.value * 2
})

// methods: 每次调用都执行
function getDoubled() {
  console.log('method 执行')
  return count.value * 2
}

// 模板中多次使用
// {{ doubled }} {{ doubled }} {{ doubled }}  → 只打印一次
// {{ getDoubled() }} {{ getDoubled() }} {{ getDoubled() }}  → 打印三次
</script>
```

### 2.4 强制重新计算
```vue
<script setup>
import { ref, computed } from 'vue'

const dep = ref(0)
const result = computed(() => dep.value * 2)

// 通过修改依赖来触发重新计算
function forceRecalculate() {
  dep.value = dep.value  // 触发依赖更新
}
</script>
```

## 三、注意事项与常见陷阱
- computed 的缓存依赖响应式系统，非响应式数据不会触发更新
- 链式 computed 中，中间 computed 变化会触发下游 computed 重新计算
- 不要在 getter 中执行副作用（如修改数据、发请求）
- computed 懒求值避免了不必要的计算，是性能优化的关键
