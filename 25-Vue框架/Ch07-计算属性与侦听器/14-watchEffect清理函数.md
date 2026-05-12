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

## 四、竞态条件详解

竞态条件是指：当依赖快速变化时，前一次异步操作的结果可能覆盖后一次的结果。

```vue
<script setup>
import { ref, watchEffect } from 'vue'

const userId = ref(1)
const userData = ref(null)
const loading = ref(false)

watchEffect(async (onInvalidate) => {
  let cancelled = false
  onInvalidate(() => { cancelled = true })

  loading.value = true

  // 模拟不同响应时间的请求
  const res = await fetch(`/api/users/${userId.value}`)
  const data = await res.json()

  // 如果这次请求已被取消（用户快速切换了 userId），不更新数据
  if (!cancelled) {
    userData.value = data
    loading.value = false
  }
})

// 模拟场景：userId 从 1 → 2 → 3 快速变化
// 请求 1（userId=1，慢）完成时，userId 已变为 3
// 如果不做取消处理，userData 会被错误地设为用户 1 的数据
</script>
```

## 五、清理函数执行时机

```js
import { ref, watchEffect } from 'vue'

const dep = ref(0)

watchEffect((onInvalidate) => {
  console.log('effect 执行, dep =', dep.value)

  onInvalidate(() => {
    console.log('清理函数执行（上一次 effect 被清理）')
  })
})

// dep.value = 1 时：
// 输出: "清理函数执行（上一次 effect 被清理）"
// 输出: "effect 执行, dep = 1"

// 停止时：
const stop = watchEffect(/* ... */)
stop()
// 输出: "清理函数执行（上一次 effect 被清理）"
```

## 六、在 Composable 中使用清理函数

```js
// composables/usePolling.js
import { ref, watchEffect } from 'vue'

export function usePolling(fetchFn, interval = 5000) {
  const data = ref(null)
  const loading = ref(false)
  const error = ref(null)

  const stop = watchEffect((onInvalidate) => {
    let cancelled = false
    onInvalidate(() => { cancelled = true })

    const poll = async () => {
      if (cancelled) return
      loading.value = true
      try {
        const result = await fetchFn()
        if (!cancelled) {
          data.value = result
          error.value = null
        }
      } catch (e) {
        if (!cancelled) error.value = e
      } finally {
        if (!cancelled) loading.value = false
      }
    }

    poll()
    const timer = setInterval(poll, interval)
    onInvalidate(() => clearInterval(timer))
  })

  return { data, loading, error, stop }
}
```
