# 多个v-model绑定

## 一、概念说明

Vue 3支持在一个组件上使用多个`v-model`，通过`v-model:名称`语法绑定不同的prop。

```vue
<!-- UserForm.vue -->
<template>
  <input :value="name" @input="$emit('update:name', $event.target.value)" />
  <input :value="email" @input="$emit('update:email', $event.target.value)" />
  <input type="checkbox" :checked="agreed" @change="$emit('update:agreed', $event.target.checked)" />
</template>

<script setup>
defineProps({ name: String, email: String, agreed: Boolean })
defineEmits(['update:name', 'update:email', 'update:agreed'])
</script>

<!-- 使用 -->
<template>
  <UserForm
    v-model:name="form.name"
    v-model:email="form.email"
    v-model:agreed="form.agreed"
  />
</template>
```

## 二、具体用法

### 使用defineModel

```vue
<!-- UserForm.vue (Vue 3.4+) -->
<script setup>
const name = defineModel('name', { default: '' })
const email = defineModel('email', { default: '' })
const agreed = defineModel('agreed', { default: false })
</script>

<template>
  <input v-model="name" />
  <input v-model="email" />
  <input type="checkbox" v-model="agreed" />
</template>
```

### 表单组件示例

```vue
<!-- AddressForm.vue -->
<script setup>
const province = defineModel('province', { default: '' })
const city = defineModel('city', { default: '' })
const detail = defineModel('detail', { default: '' })
</script>

<template>
  <select v-model="province"><option>北京</option></select>
  <select v-model="city"><option>朝阳区</option></select>
  <input v-model="detail" placeholder="详细地址" />
</template>
```

## 三、注意事项与常见陷阱

1. 默认v-model的prop名是`modelValue`，命名v-model使用指定名称
2. 每个v-model对应一个`update:xxx`事件
3. 使用defineModel简化代码，无需手动声明props和emit
4. 多个v-model适合表单组件的不同字段绑定
5. 不要过度使用，简单场景一个v-model足够
