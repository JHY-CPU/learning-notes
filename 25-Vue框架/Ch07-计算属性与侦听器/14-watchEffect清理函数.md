# watchEffect 清理函数

## 一、概念说明
`watchEffect` 的回调接收一个 `onInvalidate` 参数，用于注册**清理函数**。当 watchEffect 重新执行或被停止时，会先调用清理函数来取消上一次的副作用（如未完成的请求）。

## 二、具体用法

### 2.1 取消上一次的请求
```vue
<script setup>
import { ref, watchEffect } from 'vue'

const query = ref('')
const results = ref([])

watchEffect(async (onInvalidate) => {
  // 保存当前请求的引用
  let cancelled = false

  onInvalidate(() => {
    // 下一次执行前，取消上一次的请求
    cancelled = true
  })

  const res = await fetch(`/api/search?q=${query.value}`)

  if (!cancelled) {
    results.value = await res.json()
  }
})
</script>
```

### 2.2 使用 AbortController
```vue
<script setup>
import { ref, watchEffect } from 'vue'

const id = ref(1)
const data = ref(null)

watchEffect(async (onInvalidate) => {
  const controller = new AbortController()

  // 清理：取消上一次的请求
  onInvalidate(() => controller.abort())

  try {
    const res = await fetch(`/api/data/${id.value}`, {
      signal: controller.signal
    })
    data.value = await res.json()
  } catch (e) {
    if (e.name !== 'AbortError') throw e
  }
})
</script>
```

### 2.3 清理定时器
```vue
<script setup>
import { ref, watchEffect } from 'vue'

const interval = ref(1000)
const count = ref(0)

watchEffect((onInvalidate) => {
  const timer = setInterval(() => {
    count.value++
  }, interval.value)

  onInvalidate(() => clearInterval(timer))
})
</script>
```

## 三、注意事项与常见陷阱
- 每次 watchEffect 重新执行前都会调用上一次的清理函数
- 停止 watchEffect 时也会调用清理函数
- 常用于取消异步请求、清除定时器、移除事件监听
- 清理函数是解决竞态条件的有效方式
