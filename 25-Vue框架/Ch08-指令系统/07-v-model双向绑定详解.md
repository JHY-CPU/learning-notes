# v-model 双向绑定详解

## 一、概念说明
`v-model` 在表单元素上创建**双向数据绑定**。它根据控件类型自动选择正确的绑定方式（value/input、checked/change 等）。

## 二、具体用法

### 2.1 文本输入
```vue
<template>
  <input v-model="text" />
  <p>输入内容: {{ text }}</p>

  <!-- 等价于 -->
  <input :value="text" @input="text = $event.target.value" />
</template>
<script setup>
import { ref } from 'vue'
const text = ref('')
</script>
```

### 2.2 各种表单控件
```vue
<template>
  <!-- 多行文本 -->
  <textarea v-model="message"></textarea>

  <!-- 复选框（单个） -->
  <input type="checkbox" v-model="checked" />

  <!-- 复选框（多个） -->
  <input type="checkbox" value="Vue" v-model="frameworks" />
  <input type="checkbox" value="React" v-model="frameworks" />

  <!-- 单选框 -->
  <input type="radio" value="男" v-model="gender" />
  <input type="radio" value="女" v-model="gender" />

  <!-- 选择框 -->
  <select v-model="selected">
    <option value="a">选项A</option>
    <option value="b">选项B</option>
  </select>
</template>
<script setup>
import { ref } from 'vue'
const text = ref('')
const message = ref('')
const checked = ref(false)
const frameworks = ref([])
const gender = ref('男')
const selected = ref('a')
</script>
```

### 2.3 修饰符
```vue
<template>
  <!-- .trim 自动去除首尾空格 -->
  <input v-model.trim="username" />

  <!-- .number 自动转为数字 -->
  <input v-model.number="age" type="number" />

  <!-- .lazy 失去焦点时更新 -->
  <input v-model.lazy="searchText" />
</template>
```

### 2.4 自定义组件使用 v-model
```vue
<!-- 父组件 -->
<CustomInput v-model="value" />

<!-- 子组件 CustomInput.vue -->
<script setup>
defineProps(['modelValue'])
defineEmits(['update:modelValue'])
</script>
<template>
  <input :value="modelValue" @input="$emit('update:modelValue', $event.target.value)" />
</template>
```

## 三、注意事项与常见陷阱
- v-model 在 textarea 中不能用插值代替
- 复选框绑定数组时，value 应为字符串
- `.number` 修饰符在 `type="text"` 上无效
- 自定义组件需要实现 `modelValue` prop 和 `update:modelValue` 事件
