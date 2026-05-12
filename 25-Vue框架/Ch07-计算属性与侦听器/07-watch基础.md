# watch 基础

## 一、概念说明
`watch` 用于监听响应式数据的变化，在变化时执行**回调函数**。与 computed 不同，watch 用于执行副作用（如 API 请求、日志记录），而非返回计算值。

## 二、具体用法

### 2.1 侦听 ref
```vue
<script setup>
import { ref, watch } from 'vue'

const question = ref('')

watch(question, (newVal, oldVal) => {
  console.log(`问题从 "${oldVal}" 变为 "${newVal}"`)
})
</script>

<template>
  <input v-model="question" placeholder="输入问题" />
</template>
```

### 2.2 回调参数详解
```vue
<script setup>
import { ref, watch } from 'vue'

const count = ref(0)

watch(count, (newValue, oldValue) => {
  console.log(`新值: ${newValue}, 旧值: ${oldValue}`)
})
</script>
```

### 2.3 异步回调
```vue
<script setup>
import { ref, watch } from 'vue'

const searchQuery = ref('')

watch(searchQuery, async (query) => {
  if (!query) return
  const results = await fetch(`/api/search?q=${query}`)
    .then(r => r.json())
  console.log('搜索结果:', results)
})
</script>
```

### 2.4 立即执行 + 深度侦听
```vue
<script setup>
import { ref, watch } from 'vue'

const user = ref({ name: '张三', address: { city: '北京' } })

watch(user, (newVal) => {
  console.log('用户信息变化:', newVal)
}, { immediate: true, deep: true })
</script>
```

## 三、注意事项与常见陷阱
- 默认情况下 watch 是**懒执行**的，首次不会触发
- 使用 `{ immediate: true }` 让 watch 立即执行一次
- 侦听 ref 时回调接收 `(newVal, oldVal)`
- 侦听 reactive 对象时，oldVal 和 newVal 是同一个引用（需要 deep: true）

## 四、watch 返回值与停止

```vue
<script setup>
import { ref, watch } from 'vue'

const count = ref(0)

// watch 返回停止函数
const stop = watch(count, (val) => {
  console.log('count:', val)
  if (val >= 10) {
    stop()  // 达到 10 后停止侦听
    console.log('已停止侦听')
  }
})

// 组件卸载时自动停止，也可手动停止
function manualStop() {
  stop()
}
</script>
```

## 五、实际应用场景

### 5.1 表单自动保存
```vue
<script setup>
import { ref, watch } from 'vue'

const formData = ref({
  title: '',
  content: '',
  tags: []
})

let saveTimer = null
watch(formData, () => {
  clearTimeout(saveTimer)
  saveTimer = setTimeout(() => {
    localStorage.setItem('draft', JSON.stringify(formData.value))
    console.log('草稿已自动保存')
  }, 1000)
}, { deep: true })
</script>
```

### 5.2 路由参数变化重新请求
```vue
<script setup>
import { ref, watch } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const userData = ref(null)

watch(
  () => route.params.id,
  async (id) => {
    if (!id) return
    userData.value = await fetch(`/api/users/${id}`).then(r => r.json())
  },
  { immediate: true }
)
</script>
```

### 5.3 多条件搜索
```vue
<script setup>
import { ref, watch } from 'vue'

const keyword = ref('')
const category = ref('')
const priceRange = ref([0, 1000])

// 同时监听多个条件
watch([keyword, category, priceRange], async () => {
  const params = new URLSearchParams({
    q: keyword.value,
    cat: category.value,
    minPrice: priceRange.value[0],
    maxPrice: priceRange.value[1]
  })
  const results = await fetch(`/api/products?${params}`).then(r => r.json())
  console.log('搜索结果:', results)
})
</script>
```

## 六、常见错误

```js
// ❌ 错误：在 watch 回调中修改被监听的值（无限循环）
watch(count, (val) => {
  count.value = val + 1  // 导致无限循环！
})

// ❌ 错误：忘记处理异步竞态
watch(id, async (newId) => {
  // 如果 id 快速变化，旧请求可能比新请求晚返回
  userData.value = await fetch(`/api/users/${newId}`).then(r => r.json())
})

// ✅ 正确：使用 AbortController
watch(id, async (newId, oldId, onCleanup) => {
  const controller = new AbortController()
  onCleanup(() => controller.abort())
  try {
    userData.value = await fetch(`/api/users/${newId}`, {
      signal: controller.signal
    }).then(r => r.json())
  } catch (e) {
    if (e.name !== 'AbortError') throw e
  }
})
```
