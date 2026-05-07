# Props 单向数据流

## 一、概念说明

Vue 的 props 遵循**单向数据流**：数据只能从父组件流向子组件。子组件不应该直接修改 props，因为这会导致数据流难以追踪和调试。

当子组件需要修改数据时，应该通过**事件**通知父组件来修改。

```vue
<!-- 子组件 -->
<script setup>
const props = defineProps({
  modelValue: Number
})
const emit = defineEmits(['update:modelValue'])

// 错误: 直接修改 props
// props.modelValue++

// 正确: 通过 emit 通知父组件
function increment() {
  emit('update:modelValue', props.modelValue + 1)
}
</script>

<template>
  <span>{{ modelValue }}</span>
  <button @click="increment">+1</button>
</template>
```

## 二、具体用法

### 2.1 事件通知模式

```vue
<!-- 子组件: TodoItem.vue -->
<script setup>
const props = defineProps({
  todo: { type: Object, required: true }
})
const emit = defineEmits(['toggle', 'delete'])
</script>

<template>
  <div :class="{ done: todo.done }">
    <input type="checkbox" :checked="todo.done" @change="emit('toggle', todo.id)" />
    <span>{{ todo.text }}</span>
    <button @click="emit('delete', todo.id)">删除</button>
  </div>
</template>
```

### 2.2 内部副本模式

```vue
<script setup>
import { ref, watch } from 'vue'

const props = defineProps({ initialValue: String })
const emit = defineEmits(['update'])

// 创建内部副本
const localValue = ref(props.initialValue)

// 同步父组件的变化
watch(() => props.initialValue, (newVal) => {
  localValue.value = newVal
})

// 提交时通知父组件
function submit() {
  emit('update', localValue.value)
}
</script>
```

### 2.3 computed 模式

```vue
<script setup>
import { computed } from 'vue'

const props = defineProps({ count: Number })
const emit = defineEmits(['update:count'])

const localCount = computed({
  get: () => props.count,
  set: (val) => emit('update:count', val)
})
</script>
```

## 三、注意事项与常见陷阱

- **永远不要**直接修改 props（会产生警告和难以调试的问题）
- 对象类型的 props，子组件修改其属性不会报错但仍然不推荐
- 使用 `v-model` 是双向数据流的最佳实践
- 如果需要本地编辑，创建 props 的内部副本
- 父组件的数据变化会自动同步到子组件
