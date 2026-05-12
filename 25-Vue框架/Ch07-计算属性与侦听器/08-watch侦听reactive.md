# watch 侦听 reactive

## 一、概念说明
侦听 `reactive` 对象时有特殊行为：默认是**深度侦听**的，且回调中的 `oldValue` 和 `newValue` 是**同一个引用**。这与侦听 ref 不同。

## 二、具体用法

### 2.1 基本侦听
```vue
<script setup>
import { reactive, watch } from 'vue'

const state = reactive({
  name: '张三',
  age: 25,
  address: { city: '北京' }
})

// 自动深度侦听
watch(state, (newVal, oldVal) => {
  console.log(newVal === oldVal)  // true！同一个引用
  console.log('状态变化:', newVal)
})
</script>
```

### 2.2 获取旧值的陷阱
```vue
<script setup>
import { reactive, watch } from 'vue'

const state = reactive({ count: 0 })

watch(state, (newVal, oldVal) => {
  // ❌ oldVal 和 newVal 是同一个对象
  console.log(newVal.count === oldVal.count)  // true

  // ✅ 如果需要旧值，使用 getter 函数 + clone
  // 或者改用 ref
})
</script>
```

### 2.3 侦听 reactive 的单个属性（推荐）
```vue
<script setup>
import { reactive, watch } from 'vue'

const state = reactive({ name: '张三', age: 25 })

// 使用 getter 函数，可以获得正确的 oldVal/newVal
watch(
  () => state.name,
  (newVal, oldVal) => {
    console.log(`名字从 ${oldVal} 变为 ${newVal}`)
  }
)
</script>
```

### 2.4 侦听 reactive 数组
```vue
<script setup>
import { reactive, watch } from 'vue'

const state = reactive({ items: [1, 2, 3] })

// 侦听数组
watch(() => state.items, (newVal, oldVal) => {
  console.log('数组变化')
}, { deep: true })
</script>
```

## 三、注意事项与常见陷阱
- 侦听整个 reactive 对象时 oldVal === newVal，**无法获取旧值**
- 需要旧值时，使用 getter 函数 `watch(() => state.prop, cb)`
- reactive 对象默认深度侦听，无需加 `deep: true`
- 推荐尽量用 getter 函数而非直接侦听整个 reactive 对象

## 四、深层嵌套对象的侦听

```vue
<script setup>
import { reactive, watch } from 'vue'

const form = reactive({
  user: {
    name: '张三',
    address: {
      city: '北京',
      district: '朝阳'
    }
  }
})

// ❌ 侦听整个对象：oldVal === newVal，无法获取旧值
watch(form, (newVal, oldVal) => {
  console.log(newVal === oldVal)  // true
})

// ✅ 使用 getter 侦听特定嵌套属性
watch(
  () => form.user.address.city,
  (newCity, oldCity) => {
    console.log(`城市从 ${oldCity} 变为 ${newCity}`)
  }
)

// ✅ 使用 getter 获取浅拷贝以获取旧值
watch(
  () => ({ ...form.user }),
  (newUser, oldUser) => {
    console.log('新:', newUser, '旧:', oldUser)
  }
)
</script>
```

## 五、性能对比

```js
import { reactive, watch } from 'vue'

const largeState = reactive({
  users: Array.from({ length: 10000 }, (_, i) => ({ id: i, name: `User${i}` })),
  config: { theme: 'light', lang: 'zh' },
  metadata: { /* 大量数据 */ }
})

// ❌ 深度侦听整个大对象 — 性能差
watch(largeState, cb, { deep: true })

// ✅ 只侦听需要的属性 — 性能好
watch(() => largeState.config.theme, cb)

// ✅ 侦听数组长度变化
watch(() => largeState.users.length, cb)
```
