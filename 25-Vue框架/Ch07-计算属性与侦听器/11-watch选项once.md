# watch 选项 once

## 一、概念说明
`once` 选项让 watch 回调**只触发一次**后自动停止侦听。这在只需要响应首次变化的场景中非常有用。

## 二、具体用法

### 2.1 基本用法
```vue
<script setup>
import { ref, watch } from 'vue'

const isFirstVisit = ref(false)

// 只在首次变为 true 时触发
watch(isFirstVisit, (val) => {
  if (val) {
    console.log('用户首次访问')
    showWelcomeModal()
  }
}, { once: true })
</script>
```

### 2.2 等待数据加载完成
```vue
<script setup>
import { ref, watch } from 'vue'

const data = ref(null)

// 数据首次加载后执行初始化，之后不再触发
watch(data, (val) => {
  if (val) {
    initChart(val)
    startAnimation()
  }
}, { once: true })
</script>
```

### 2.3 once 与 immediate 组合
```vue
<script setup>
import { ref, watch } from 'vue'

const user = ref(null)

// 立即检查一次，之后不再侦听
watch(user, (val) => {
  if (val) {
    trackLogin(val.id)
  }
}, { immediate: true, once: true })
</script>
```

### 2.4 等效的手动停止写法
```vue
<script setup>
import { ref, watch } from 'vue'

const value = ref(0)

// Vue 3.4 之前的等效写法
const stop = watch(value, (val) => {
  console.log('值变为:', val)
  stop()  // 手动停止
})
</script>
```

## 三、注意事项与常见陷阱
- `once` 是 Vue 3.4+ 的特性，旧版本使用手动 `stop()`
- 与 `immediate` 组合时，首次执行（含创建时）就停止
- 适合一次性初始化、首次提示等场景
- 一旦停止无法重新激活，需要重新创建 watch

## 五、once 的实际场景

### 5.1 首次数据加载后的初始化
```vue
<script setup>
import { ref, watch } from 'vue'

const chartData = ref(null)

// 数据首次加载后初始化图表，之后数据更新由图表库自己处理
watch(chartData, (data) => {
  if (data) {
    initChart(data)
    // 图表库内部处理后续数据更新
  }
}, { once: true })
</script>
```

### 5.2 用户交互触发一次性操作
```vue
<script setup>
import { ref, watch } from 'vue'

const hasScrolled = ref(false)
const hasClicked = ref(false)

// 用户首次滚动时显示引导
watch(hasScrolled, (val) => {
  if (val) showScrollHint()
}, { once: true })

// 用户首次点击时隐藏引导
watch(hasClicked, (val) => {
  if (val) dismissTour()
}, { once: true })
</script>
```

### 5.3 与 watchEffect 的对比
```js
// watch + once：可以获取 oldVal
watch(source, (newVal, oldVal) => {
  init(newVal)
}, { once: true })

// watchEffect + 手动 stop：没有 oldVal
const stop = watchEffect(() => {
  if (condition.value) {
    doSomething(source.value)
    stop()
  }
})
```
