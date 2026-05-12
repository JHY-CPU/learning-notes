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

## 四、使用唯一ID管理动态项

```vue
<script setup>
import { reactive, ref } from 'vue'

let nextId = 1
const items = reactive([
  { id: nextId++, name: '', quantity: 1, price: 0 }
])

const addItem = () => {
  items.push({ id: nextId++, name: '', quantity: 1, price: 0 })
}

const removeItem = (id) => {
  const index = items.findIndex(item => item.id === id)
  if (index > -1 && items.length > 1) {
    items.splice(index, 1)
  }
}
</script>

<template>
  <div v-for="item in items" :key="item.id" class="form-row">
    <input v-model="item.name" placeholder="商品名" />
    <input v-model.number="item.quantity" type="number" min="1" />
    <input v-model.number="item.price" type="number" min="0" />
    <button @click="removeItem(item.id)" :disabled="items.length <= 1">删除</button>
  </div>
</template>
```

## 五、拖拽排序

```vue
<script setup>
import { ref } from 'vue'

const items = ref([
  { id: 1, text: '项目A' },
  { id: 2, text: '项目B' },
  { id: 3, text: '项目C' }
])

let draggedIndex = null

const onDragStart = (index) => { draggedIndex = index }
const onDrop = (index) => {
  if (draggedIndex === null) return
  const [item] = items.value.splice(draggedIndex, 1)
  items.value.splice(index, 0, item)
  draggedIndex = null
}
</script>

<template>
  <div
    v-for="(item, index) in items"
    :key="item.id"
    draggable="true"
    @dragstart="onDragStart(index)"
    @dragover.prevent
    @drop="onDrop(index)"
  >
    {{ item.text }}
  </div>
</template>
```

## 六、汇总计算

```vue
<script setup>
import { reactive, computed } from 'vue'

const items = reactive([{ name: '', quantity: 1, price: 0 }])

const totalAmount = computed(() =>
  items.reduce((sum, item) => sum + item.quantity * item.price, 0)
)

const itemCount = computed(() => items.length)
</script>

<template>
  <div v-for="(item, index) in items" :key="index">
    <input v-model="item.name" />
    <input v-model.number="item.quantity" type="number" />
    <input v-model.number="item.price" type="number" />
    <span>小计: {{ item.quantity * item.price }}</span>
  </div>
  <div class="summary">
    <p>共 {{ itemCount }} 项，合计: {{ totalAmount }}</p>
  </div>
</template>
```

## 三、注意事项与常见陷阱

1. 使用`reactive`数组管理动态项
2. 删除时保留至少一项，避免空表单
3. `:key`使用index时注意与splice的兼容（用唯一ID更好）
4. 验证需要遍历数组中的每一项
5. 提交时需要深拷贝（避免引用问题）
6. 唯一ID推荐使用`ref`计数器或`crypto.randomUUID()`
7. 拖拽排序时注意更新DOM key以触发正确重渲染
