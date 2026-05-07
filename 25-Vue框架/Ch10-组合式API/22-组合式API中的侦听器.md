# 组合式API中的侦听器

## 一、概念说明

组合式API提供两个侦听器函数：`watch`（明确指定数据源）和`watchEffect`（自动追踪依赖）。它们在setup中使用，替代选项式API的watch选项。

```vue
<script setup>
import { ref, watch, watchEffect } from 'vue'

const keyword = ref('')
const results = ref([])

// watch: 明确监听keyword
watch(keyword, async (newVal) => {
  if (newVal) {
    results.value = await search(newVal)
  }
}, { immediate: true })

// watchEffect: 自动追踪keyword
watchEffect(async () => {
  if (keyword.value) {
    results.value = await search(keyword.value)
  }
})
</script>
```

## 二、具体用法

### watch的三种形式

```js
import { ref, reactive, watch } from 'vue'

// 1. 监听ref
const count = ref(0)
watch(count, (newVal, oldVal) => {
  console.log(`${oldVal} -> ${newVal}`)
})

// 2. 监听getter函数
const state = reactive({ count: 0 })
watch(() => state.count, (newVal) => {
  console.log('count:', newVal)
})

// 3. 监听数组（多数据源）
watch([count, () => state.count], ([c1, c2]) => {
  console.log(c1, c2)
})
```

### watch选项

```js
watch(source, callback, {
  immediate: true,   // 立即执行一次
  deep: true,        // 深度监听
  flush: 'post'      // 'pre' | 'post' | 'sync'
})
```

### watchEffect用法

```js
const count = ref(0)

// 自动追踪count，count变化时重新执行
const stop = watchEffect(() => {
  console.log('count:', count.value)
})

// 停止侦听
stop()

// onInvalidate: 清理副作用
watchEffect((onInvalidate) => {
  const id = fetch(url)
  onInvalidate(() => cancel(id))
})
```

## 三、注意事项与常见陷阱

1. `watch`惰性执行（除非`immediate: true`），`watchEffect`立即执行
2. 监听reactive对象时需用getter：`watch(() => state.obj, ...)`
3. `watchEffect`不能获取旧值，需要旧值时用`watch`
4. 在`onUnmounted`前自动停止，也可以手动调用返回的停止函数
5. `flush: 'pre'`（默认）在DOM更新前触发，`'post'`在DOM更新后触发
