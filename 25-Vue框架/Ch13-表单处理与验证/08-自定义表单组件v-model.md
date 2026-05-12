# 自定义表单组件v-model

## 一、概念说明

`defineModel`是Vue 3.4+的宏，简化自定义组件的`v-model`实现。它自动处理props和emit。

```vue
<!-- CustomInput.vue -->
<template>
  <input :value="modelValue" @input="modelValue = $event.target.value" />
</template>

<script setup>
// Vue 3.4+ 简洁写法
const modelValue = defineModel()
</script>

<!-- 使用 -->
<template>
  <CustomInput v-model="text" />
</template>
```

## 二、具体用法

### 带参数的defineModel

```vue
<!-- 自定义组件 -->
<script setup>
const modelValue = defineModel({ type: String, default: '' })
const count = defineModel('count', { type: Number, default: 0 })
</script>

<!-- 使用 -->
<template>
  <MyComponent v-model="name" v-model:count="num" />
</template>
```

### 自定义修饰符

```vue
<script setup>
const [modelValue, modifiers] = defineModel({
  set(value) {
    // 处理.trim修饰符
    if (modifiers.trim) return value.trim()
    return value
  }
})
</script>

<!-- 使用 -->
<template>
  <CustomInput v-model.trim="text" />
</template>
```

### 手动实现（兼容3.4之前）

```vue
<script setup>
const props = defineProps({ modelValue: String })
const emit = defineEmits(['update:modelValue'])

const onInput = (e) => {
  emit('update:modelValue', e.target.value)
}
</script>

<template>
  <input :value="props.modelValue" @input="onInput" />
</template>
```

## 四、自定义数字输入组件

```vue
<!-- NumberInput.vue -->
<template>
  <div class="number-input">
    <button @click="decrease" :disabled="modelValue <= min">-</button>
    <input
      type="number"
      :value="modelValue"
      @input="onInput"
      :min="min"
      :max="max"
    />
    <button @click="increase" :disabled="modelValue >= max">+</button>
  </div>
</template>

<script setup>
const modelValue = defineModel({ type: Number, default: 0 })
const props = defineProps({
  min: { type: Number, default: 0 },
  max: { type: Number, default: 100 },
  step: { type: Number, default: 1 }
})

const onInput = (e) => {
  const val = Number(e.target.value)
  modelValue.value = Math.min(Math.max(val, props.min), props.max)
}
const decrease = () => { modelValue.value = Math.max(modelValue.value - props.step, props.min) }
const increase = () => { modelValue.value = Math.min(modelValue.value + props.step, props.max) }
</script>

<!-- 使用 -->
<template>
  <NumberInput v-model="quantity" :min="1" :max="99" />
</template>
```

## 五、自定义搜索选择组件

```vue
<!-- SearchSelect.vue -->
<template>
  <div class="search-select">
    <input
      v-model="searchText"
      :placeholder="placeholder"
      @focus="showDropdown = true"
      @blur="hideDropdown"
    />
    <ul v-show="showDropdown" class="dropdown">
      <li
        v-for="item in filteredOptions"
        :key="item.value"
        @mousedown="select(item)"
      >
        {{ item.label }}
      </li>
    </ul>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const modelValue = defineModel({ type: [String, Number], default: '' })
const props = defineProps({
  options: { type: Array, default: () => [] },
  placeholder: { type: String, default: '搜索...' }
})

const searchText = ref('')
const showDropdown = ref(false)

const filteredOptions = computed(() =>
  props.options.filter(o =>
    o.label.toLowerCase().includes(searchText.value.toLowerCase())
  )
)

const select = (item) => {
  modelValue.value = item.value
  searchText.value = item.label
  showDropdown.value = false
}
</script>
```

## 六、自定义验证包装组件

```vue
<!-- ValidatedInput.vue -->
<template>
  <div class="validated-field" :class="{ invalid: !!error }">
    <label v-if="label">{{ label }}</label>
    <input
      :value="modelValue"
      @input="handleInput"
      @blur="validate"
      :type="type"
      :placeholder="placeholder"
    />
    <span v-if="error" class="error-text">{{ error }}</span>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const modelValue = defineModel({ type: String, default: '' })
const props = defineProps({
  label: String,
  type: { type: String, default: 'text' },
  placeholder: String,
  rules: { type: Array, default: () => [] }
})

const error = ref('')

const handleInput = (e) => {
  modelValue.value = e.target.value
  if (error.value) validate()
}

const validate = () => {
  for (const rule of props.rules) {
    const result = rule(modelValue.value)
    if (result !== true) {
      error.value = result
      return false
    }
  }
  error.value = ''
  return true
}

defineExpose({ validate })
</script>

<!-- 使用 -->
<template>
  <ValidatedInput
    v-model="email"
    label="邮箱"
    type="email"
    :rules="[v => !!v || '必填', v => /\S+@\S+\.\S+/.test(v) || '格式不正确']"
  />
</template>
```

## 三、注意事项与常见陷阱

1. `defineModel`是Vue 3.4+特性，旧版本需手动实现
2. `defineModel`返回的是ref，在模板中自动解包
3. 多个v-model用不同名称（`v-model:xxx`）
4. 自定义组件的v-model默认prop是`modelValue`
5. 修饰符通过defineModel的第二个返回值获取
6. 复杂表单组件建议拆分为原子组件，每个组件只处理一个输入类型
7. 自定义组件暴露`validate`方法，便于外部调用验证
