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
