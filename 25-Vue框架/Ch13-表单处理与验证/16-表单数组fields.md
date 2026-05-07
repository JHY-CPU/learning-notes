# 表单数组fields

## 一、概念说明

动态增删表单项，常见于批量输入、多地址、多联系方式等场景。使用reactive数组管理动态字段。

```vue
<template>
  <form @submit.prevent="handleSubmit">
    <div v-for="(item, index) in items" :key="index" class="form-item">
      <input v-model="item.name" placeholder="商品名" />
      <input v-model.number="item.quantity" type="number" placeholder="数量" />
      <input v-model.number="item.price" type="number" placeholder="单价" />
      <button type="button" @click="removeItem(index)">删除</button>
    </div>
    <button type="button" @click="addItem">添加商品</button>
    <button type="submit">提交</button>
  </form>
</template>

<script setup>
import { reactive } from 'vue'

const items = reactive([
  { name: '', quantity: 1, price: 0 }
])

const addItem = () => {
  items.push({ name: '', quantity: 1, price: 0 })
}

const removeItem = (index) => {
  if (items.length > 1) items.splice(index, 1)
}

const handleSubmit = () => {
  console.log('提交:', JSON.parse(JSON.stringify(items)))
}
</script>
```

## 二、具体用法

### 带验证的动态表单

```vue
<script setup>
import { reactive, computed } from 'vue'

const items = reactive([{ name: '', phone: '' }])
const errors = reactive({})

const validate = () => {
  items.forEach((item, i) => {
    errors[i] = {}
    if (!item.name) errors[i].name = '姓名必填'
    if (!/^1[3-9]\d{9}$/.test(item.phone)) errors[i].phone = '手机号不正确'
  })
  return !Object.values(errors).some(e => Object.keys(e).length)
}
</script>
```

### 限制数量

```js
const MAX_ITEMS = 5
const addItem = () => {
  if (items.length < MAX_ITEMS) {
    items.push({ name: '', value: '' })
  }
}
```

## 三、注意事项与常见陷阱

1. 使用`reactive`数组管理动态项
2. 删除时保留至少一项，避免空表单
3. `:key`使用index时注意与splice的兼容（用唯一ID更好）
4. 验证需要遍历数组中的每一项
5. 提交时需要深拷贝（避免引用问题）
