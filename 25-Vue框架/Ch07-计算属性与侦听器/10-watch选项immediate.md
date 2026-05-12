# watch 选项 immediate

## 一、概念说明
默认情况下 watch 是**懒执行**的——只在数据变化后才调用回调。加上 `{ immediate: true }` 后，watch 会在**创建时立即执行一次**回调。

## 二、具体用法

### 2.1 基本用法
```vue
<script setup>
import { ref, watch } from 'vue'

const userId = ref(1)

// immediate: 注册时立即执行一次（用当前值）
watch(userId, async (id) => {
  const user = await fetch(`/api/users/${id}`).then(r => r.json())
  console.log('用户信息:', user)
}, { immediate: true })
</script>
```

### 2.2 初始化数据请求
```vue
<script setup>
import { ref, watch } from 'vue'

const page = ref(1)
const data = ref([])

watch(page, async (p) => {
  data.value = await fetch(`/api/list?page=${p}`).then(r => r.json())
}, { immediate: true })  // 组件创建时立即请求第一页
</script>
```

### 2.3 immediate + deep 组合
```vue
<script setup>
import { ref, watch } from 'vue'

const settings = ref({ theme: 'light', lang: 'zh' })

watch(settings, (val) => {
  applySettings(val)
}, { immediate: true, deep: true })  // 立即执行 + 深度侦听
</script>
```

### 2.4 immediate 时的回调参数
```vue
<script setup>
import { ref, watch } from 'vue'

const count = ref(10)

watch(count, (newVal, oldVal) => {
  console.log('new:', newVal, 'old:', oldVal)
  // immediate 首次执行时: new=10, old=undefined
}, { immediate: true })
</script>
```

## 三、注意事项与常见陷阱
- `immediate` 首次执行时 `oldValue` 是 `undefined`
- 需要初始化数据时用 `immediate` 比在 `onMounted` 中手动调用更简洁
- `immediate` 和 `watchEffect` 的区别：watchEffect 没有 oldVal
- 不要过度依赖 `immediate`，有时 onMounted 更直观

## 五、immediate vs 其他初始化方式

```vue
<script setup>
import { ref, watch, onMounted } from 'vue'

const userId = ref(1)
const userData = ref(null)

// 方式一：watch + immediate
// 优点：数据变化时自动重新请求
// 缺点：首次执行时 oldVal 是 undefined
watch(userId, async (id) => {
  userData.value = await fetch(`/api/users/${id}`).then(r => r.json())
}, { immediate: true })

// 方式二：onMounted + watch
// 优点：更清晰地分离初始化和响应逻辑
// 缺点：代码稍多
onMounted(async () => {
  userData.value = await fetch(`/api/users/${userId.value}`).then(r => r.json())
})
watch(userId, async (id) => {
  userData.value = await fetch(`/api/users/${id}`).then(r => r.json())
})

// 方式三：watchEffect
// 优点：最简洁
// 缺点：无法获取 oldVal
import { watchEffect } from 'vue'
watchEffect(async () => {
  userData.value = await fetch(`/api/users/${userId.value}`).then(r => r.json())
})
</script>
```

| 方式 | 代码量 | oldVal | 自动追踪 | 初始加载 |
|------|--------|--------|----------|----------|
| watch + immediate | 中 | undefined 首次 | 手动指定 | 是 |
| onMounted + watch | 多 | 正常 | 手动指定 | 手动 |
| watchEffect | 少 | 无 | 自动 | 是 |
