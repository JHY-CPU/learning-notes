# 透传 $attrs 通信

## 一、概念说明
`$attrs` 包含了父组件传递给子组件的**所有未被 props 声明接收的属性**。通过 `$attrs` 可以将父组件的属性透传到子组件的根元素或任意子元素上，适合封装高阶包装组件。

## 二、具体用法

### 2.1 基本用法
```vue
<!-- 父组件 -->
<MyInput placeholder="请输入" class="large" maxlength="20" />
```

```vue
<!-- MyInput.vue -->
<template>
  <input :value="modelValue" @input="$emit('update:modelValue', $event.target.value)" />
</template>
<script setup>
defineProps<{ modelValue: string }>()
defineEmits(['update:modelValue'])
</script>
```

### 2.2 禁用自动继承
```vue
<script setup>
defineOptions({ inheritAttrs: false })
</script>

<template>
  <div class="wrapper">
    <input v-bind="$attrs" />
  </div>
</template>
```

### 2.3 使用 useAttrs
```vue
<script setup>
import { useAttrs } from 'vue'
const attrs = useAttrs()
// attrs 包含属性和事件监听器
console.log(attrs.placeholder)
console.log(attrs.onClick)
</script>
```

### 2.4 组合 attrs 和 props
```vue
<script setup>
defineOptions({ inheritAttrs: false })

defineProps({
  type: { type: String, default: 'text' }
})

const attrs = useAttrs()
</script>

<template>
  <div class="input-wrapper">
    <input :type="type" v-bind="attrs" />
  </div>
</template>
```

## 三、常见用例

| 场景 | 说明 |
|------|------|
| 封装表单组件 | 透传 placeholder、maxlength 等 |
| 包装第三方组件 | 透传原生属性 |
| 高阶组件 | 动态绑定属性到子元素 |

## 四、注意事项与常见陷阱

- `$attrs` 包含属性**和**事件监听器
- class 和 style 会自动与根元素的合并
- 多根节点组件不会自动继承，需手动 `v-bind="$attrs"`
- 使用 `defineOptions({ inheritAttrs: false })` 可禁用自动继承
- `useAttrs()` 在模板中是响应式的，但在 script 中不是
- props 中已声明的属性不会出现在 `$attrs` 中

## 五、完整包装组件示例

```vue
<!-- 封装一个增强版 Input -->
<!-- EnhancedInput.vue -->
<script setup>
import { useAttrs } from 'vue'

defineOptions({ inheritAttrs: false })

defineProps({
  label: String,
  modelValue: [String, Number]
})

defineEmits(['update:modelValue'])

const attrs = useAttrs()
</script>

<template>
  <div class="input-group">
    <label v-if="label">{{ label }}</label>
    <input
      :value="modelValue"
      v-bind="attrs"
      @input="$emit('update:modelValue', $event.target.value)"
    />
    <span v-if="attrs.maxlength" class="char-count">
      {{ (modelValue || '').length }} / {{ attrs.maxlength }}
    </span>
  </div>
</template>

<!-- 使用 -->
<EnhancedInput
  v-model="name"
  label="姓名"
  placeholder="请输入姓名"
  maxlength="20"
  class="large"
/>
```

## 六、多根节点的 attrs 继承

```vue
<!-- 多根节点组件不会自动继承 attrs -->
<!-- 必须手动绑定到某个元素 -->
<template>
  <label>{{ label }}</label>
  <input v-bind="$attrs" :value="modelValue" />
  <!-- 如果不写 v-bind="$attrs"，父组件传入的 class/style 不会自动应用 -->
</template>
```

## 七、attrs vs props 对比

| 特性 | props | $attrs |
| --- | --- | --- |
| 声明方式 | defineProps | 自动收集未声明的 |
| 类型检查 | 是 | 否 |
| 默认值 | 支持 | 不支持 |
| 包含事件 | 否 | 是 |
| 模板响应式 | 是 | 否（script 中） |
| 传递给子元素 | 手动 | v-bind="$attrs" |

## 八、实际应用场景

- **UI 库封装：**基础组件透传原生属性（placeholder、disabled 等）
- **高阶组件包装：**在原组件基础上增加功能，保留原组件 API
- **表单封装：**透传 type、required、pattern 等 HTML 属性
- **响应式封装：**对外暴露简单 props，内部通过 $attrs 处理复杂属性
