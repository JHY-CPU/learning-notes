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
