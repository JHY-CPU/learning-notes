# `<script setup>`语法糖

## 一、概念说明

`<script setup>`是Vue 3.2+引入的编译时语法糖，用于在SFC（单文件组件）中使用组合式API。它极大简化了代码，无需手动导出或注册。

```vue
<template>
  <p>{{ message }}</p>
  <button @click="greet">问候</button>
</template>

<script setup>
import { ref } from 'vue'

const message = ref('你好，Vue 3！')

function greet() {
  alert(message.value)
}
</script>
```

## 二、具体用法

### 自动暴露
`<script setup>`中的顶层绑定（变量、函数、import）自动暴露给模板：

```vue
<script setup>
import MyComponent from './MyComponent.vue'
import { ref } from 'vue'

const count = ref(0)       // 可在模板使用
const add = () => count.value++  // 可在模板使用
</script>
```

### 使用defineProps和defineEmits

```vue
<script setup>
// 定义props
const props = defineProps({
  title: { type: String, required: true },
  count: { type: Number, default: 0 }
})

// 定义事件
const emit = defineEmits(['update', 'delete'])

function handleUpdate() {
  emit('update', props.title)
}
</script>
```

### 使用defineExpose暴露方法

```vue
<script setup>
import { ref } from 'vue'

const count = ref(0)
const reset = () => { count.value = 0 }

// 父组件通过ref访问
defineExpose({ count, reset })
</script>
```

## 三、注意事项与常见陷阱

1. `<script setup>`中每个绑定都是`setup()`的返回值，无需手动`return`
2. `defineProps`、`defineEmits`、`defineExpose`是编译宏，无需import
3. 可以与普通`<script>`共存（用于声明name、inheritAttrs等）
4. 默认情况下，`<script setup>`中的顶层变量不是全局的，仅当前组件可用
5. 自动注册组件，无需在`components`选项中声明
