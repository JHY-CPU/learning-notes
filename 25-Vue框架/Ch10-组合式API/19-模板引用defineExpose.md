# 模板引用defineExpose

## 一、概念说明

`defineExpose`用于在`<script setup>`中显式暴露组件的属性和方法给父组件。默认情况下`<script setup>`的绑定是私有的，需要显式暴露。

```vue
<!-- ChildComponent.vue -->
<template>
  <div>
    <p>内部计数: {{ count }}</p>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const count = ref(0)
const reset = () => { count.value = 0 }
const add = (n) => { count.value += n }

// 暴露给父组件
defineExpose({ count, reset, add })
</script>
```

```vue
<!-- ParentComponent.vue -->
<template>
  <ChildComponent ref="childRef" />
  <button @click="childRef?.reset()">重置子组件</button>
</template>

<script setup>
import { ref } from 'vue'
import ChildComponent from './ChildComponent.vue'

const childRef = ref(null)
</script>
```

## 二、具体用法

### 暴露方法和状态

```vue
<script setup>
import { ref, computed } from 'vue'

const formData = ref({ name: '', email: '' })
const isValid = computed(() => formData.value.name && formData.value.email)

const validate = () => isValid.value
const reset = () => {
  formData.value = { name: '', email: '' }
}
const getData = () => ({ ...formData.value })

defineExpose({
  validate,
  reset,
  getData,
  formData  // 暴露响应式数据
})
</script>
```

## 三、注意事项与常见陷阱

1. 在`<script setup>`中，**只有被`defineExpose`暴露的内容**父组件才能访问
2. 普通`<script>`中setup返回的所有内容自动暴露，无需defineExpose
3. 暴露`ref`时父组件通过`.value`访问
4. 避免暴露过多内部实现，保持组件封装性
5. 父组件访问子组件ref时，需要在`onMounted`之后才可用

## 四、defineExpose 的最佳实践

### 4.1 只暴露公共 API
```vue
<script setup>
import { ref, computed } from 'vue'

// 内部状态（不暴露）
const formData = ref({ name: '', email: '', password: '' })
const internalErrors = ref({})

// 内部方法（不暴露）
function validateField(field) { /* ... */ }

// 公共方法（暴露）
function validate() {
  return Object.keys(internalErrors.value).length === 0
}
function reset() {
  formData.value = { name: '', email: '', password: '' }
  internalErrors.value = {}
}
function submit() {
  if (validate()) {
    emit('submit', { ...formData.value })
  }
}

// 只暴露公共 API，保持封装性
defineExpose({ validate, reset, submit })
</script>
```

### 4.2 异步方法暴露
```vue
<script setup>
import { ref } from 'vue'

const loading = ref(false)

async function refresh() {
  loading.value = true
  try {
    // 异步操作
  } finally {
    loading.value = false
  }
}

defineExpose({ refresh, loading })
</script>
```

```vue
<!-- 父组件使用 -->
<template>
  <DataList ref="listRef" />
  <button @click="handleRefresh">刷新</button>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const listRef = ref(null)

onMounted(async () => {
  // 调用子组件方法
  await listRef.value?.refresh()
})
</script>
```

## 五、defineExpose vs expose（setup context）

| 特性 | defineExpose | expose |
|------|-------------|--------|
| 使用场景 | `<script setup>` | 普通 `setup()` |
| 语法 | 编译宏 | 函数调用 |
| TypeScript | 完整支持 | 部分支持 |
| 默认行为 | 默认私有 | 默认公开 |
