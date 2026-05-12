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

## 四、`<script setup>` 的进阶用法

### 4.1 defineProps 的 TypeScript 写法

```vue
<script setup lang="ts">
// 运行时声明
const props = defineProps({
  title: { type: String, required: true },
  count: { type: Number, default: 0 }
})

// 基于类型的声明（Vue 3.2+）
interface Props {
  title: string
  count?: number
  items?: string[]
}
const props = defineProps<Props>()

// 带默认值（Vue 3.3+）
const props = withDefaults(defineProps<Props>(), {
  count: 0,
  items: () => []
})
</script>
```

### 4.2 defineEmits 的 TypeScript 写法

```vue
<script setup lang="ts">
// 运行时声明
const emit = defineEmits(['update', 'delete'])

// 基于类型的声明
const emit = defineEmits<{
  update: [id: number, value: string]
  delete: [id: number]
}>()

// 使用
function handleUpdate() {
  emit('update', 1, 'new value')
}
</script>
```

### 4.3 defineModel（Vue 3.4+）

```vue
<script setup>
// 自动定义 modelValue prop 和 update:modelValue 事件
const modelValue = defineModel()

// 带类型和默认值
const modelValue = defineModel({ default: '' })

// 多个 v-model
const title = defineModel('title')
const count = defineModel('count', { default: 0 })
</script>
```

### 4.4 与普通 `<script>` 共存

```vue
<!-- 声明组件名和 inheritAttrs -->
<script lang="ts">
export default {
  name: 'MyComponent',
  inheritAttrs: false
}
</script>

<script setup lang="ts">
import { ref } from 'vue'
const count = ref(0)
</script>
```

### 4.5 useSlots 和 useAttrs

```vue
<script setup>
import { useSlots, useAttrs } from 'vue'

const slots = useSlots()
const attrs = useAttrs()

// 检查插槽是否存在
const hasHeader = computed(() => !!slots.header)

// 非 prop 的 attribute
console.log(attrs.class, attrs.id)
</script>
```

## 五、`<script setup>` 的限制

| 限制 | 解决方案 |
|------|---------|
| 不能动态组件名 | 使用 `defineOptions`（3.3+）|
| 不能声明额外选项 | 与普通 `<script>` 共存 |
| 没有 this | 使用 context 参数 |
| 默认私有暴露 | `defineExpose` 显式暴露 |
