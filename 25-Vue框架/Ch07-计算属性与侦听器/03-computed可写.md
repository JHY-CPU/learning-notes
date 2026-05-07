# computed 可写

## 一、概念说明
默认的 computed 是只读的（只有 getter）。Vue 也支持**可写计算属性**，通过同时提供 getter 和 setter 来实现双向计算。

## 二、具体用法

### 2.1 getter/setter 写法
```vue
<script setup>
import { ref, computed } from 'vue'

const firstName = ref('张')
const lastName = ref('三')

const fullName = computed({
  // getter: 拼接姓名
  get() {
    return `${firstName.value} ${lastName.value}`
  },
  // setter: 拆分姓名
  set(newValue) {
    const [first, last] = newValue.split(' ')
    firstName.value = first
    lastName.value = last
  }
})
</script>

<template>
  <p>姓: {{ firstName }}，名: {{ lastName }}</p>
  <p>全名: {{ fullName }}</p>
  <input v-model="fullName" placeholder="输入全名" />
  <!-- 输入 "李 四" → firstName 变为 "李"，lastName 变为 "四" -->
</template>
```

### 2.2 表单位置转换
```vue
<script setup>
import { ref, computed } from 'vue'

const x = ref(10)
const y = ref(20)

const position = computed({
  get: () => `${x.value},${y.value}`,
  set: (val) => {
    const [newX, newY] = val.split(',').map(Number)
    x.value = newX
    y.value = newY
  }
})
</script>
```

### 2.3 限制值范围
```vue
<script setup>
import { ref, computed } from 'vue'

const rawValue = ref(50)

const percentage = computed({
  get: () => rawValue.value,
  set: (val) => {
    rawValue.value = Math.max(0, Math.min(100, val))  // 限制 0-100
  }
})
</script>
```

## 三、注意事项与常见陷阱
- 只需要读取值时使用简写形式（只传 getter 函数）
- setter 中不要触发无限循环（如在 setter 中修改自身依赖）
- 可写 computed 适合表单数据转换场景
- 大多数场景下用 ref + watch 比可写 computed 更清晰
