# $parent 与 $refs

## 一、概念说明
`$parent` 和 `$refs` 是选项式 API 中访问组件实例的方式。`$parent` 指向父组件实例，`$refs` 包含所有通过 `ref` 注册的子组件/DOM 元素。

## 二、具体用法

### 2.1 $parent 访问父组件（选项式 API）
```vue
<script>
export default {
  methods: {
    callParentMethod() {
      this.$parent.parentMethod()
    }
  }
}
</script>
```

### 2.2 $refs 访问子元素（选项式 API）
```vue
<template>
  <input ref="nameInput" />
  <ChildForm ref="childForm" />
</template>
<script>
export default {
  mounted() {
    this.$refs.nameInput.focus()
    this.$refs.childForm.validate()
  }
}
</script>
```

### 2.3 组合式 API 中的等价写法
```vue
<script setup>
import { ref, onMounted } from 'vue'

const nameInput = ref(null)
const childForm = ref(null)

onMounted(() => {
  nameInput.value.focus()
  childForm.value.validate()
})
</script>
```

### 2.4 $refs 在组合式 API 中的替代方案

```vue
<!-- 组合式 API 不再使用 $refs -->
<!-- ref() 变量名直接对应模板中的 ref 属性 -->
<script setup>
import { ref, onMounted } from 'vue'

// 变量名与模板 ref 名一致
const myInput = ref(null)

onMounted(() => {
  // 直接通过 .value 访问
  myInput.value?.focus()
})
</script>

<template>
  <input ref="myInput" />
</template>
```

## 三、常见用例

| 选项式 API | 组合式 API |
|-----------|-----------|
| `this.$refs.xxx` | `xxx.value` |
| `this.$parent` | props/emit |
| `this.$children` | 不推荐 |

## 四、注意事项与常见陷阱

- 在 `<script setup>` 中 `$parent` 不可用，需通过 props/emit 替代
- `$refs` 只在组件渲染完成后可用（`onMounted` 之后）
- `$refs` 不是响应式的，不应在模板中依赖其变化
- 组合式 API 中使用 `ref()` 变量名即可替代 `$refs.xxx`
- 避免在组合式 API 中使用 `$parent`，破坏了组件独立性
- Vue 3 中 `$children` 已被移除，改用模板 ref

## 五、ref 数组处理

```vue
<template>
  <div v-for="item in items" :key="item.id">
    <ItemCard :ref="el => setItemRef(el, item.id)" :data="item" />
  </div>
</template>
<script setup>
import { ref, onBeforeUpdate } from 'vue'

const items = ref([{ id: 1 }, { id: 2 }, { id: 3 }])
const itemRefs = {}

function setItemRef(el, id) {
  if (el) itemRefs[id] = el
  else delete itemRefs[id]
}

// 重要：更新前清理旧的 ref
onBeforeUpdate(() => {
  Object.keys(itemRefs).forEach(k => delete itemRefs[k])
})

function validateAll() {
  for (const id in itemRefs) {
    itemRefs[id].validate()
  }
}
</script>
```

## 六、expose 控制访问

```vue
<!-- 子组件通过 expose 控制父组件能访问的内容 -->
<script setup>
import { ref } from 'vue'

const count = ref(0)
const secret = ref('hidden')

function increment() { count.value++ }

// 只暴露 increment，不暴露 count 和 secret
defineExpose({ increment })
</script>

<!-- 父组件 -->
<script setup>
import { ref, onMounted } from 'vue'
const childRef = ref(null)

onMounted(() => {
  childRef.value.increment()  // 可以
  // childRef.value.count     // TypeScript 中报错
  // childRef.value.secret    // 不可访问
})
</script>
```

## 七、实际应用场景

| 场景 | 推荐方式 | 原因 |
| --- | --- | --- |
| 表单验证 | 模板 ref | 调用子组件 validate 方法 |
| 滚动到子元素 | 模板 ref | DOM 操作 |
| 访问父组件方法 | props/emit | 松耦合 |
| 兄弟组件通信 | Pinia | 无需通过共同父组件 |
| 动态组件方法调用 | 模板 ref + defineExpose | 精确控制暴露内容 |
