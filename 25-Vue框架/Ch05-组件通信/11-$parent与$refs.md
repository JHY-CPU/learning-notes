# $parent 与 $refs

## 一、概念说明
`$parent` 和 `$refs` 是选项式 API 中访问组件实例的方式。`$parent` 指向父组件实例，`$refs` 包含所有通过 `ref` 注册的子组件/DOM 元素。

## 二、具体用法

### 2.1 $parent 访问父组件
```vue
<!-- 子组件（选项式 API） -->
<script>
export default {
  methods: {
    callParentMethod() {
      this.$parent.parentMethod()
    },
    getParentData() {
      return this.$parent.parentData
    }
  }
}
</script>
```

### 2.2 $refs 访问子元素
```vue
<!-- 选项式 API -->
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

## 三、注意事项与常见陷阱
- 在 `<script setup>` 中 `$parent` 不可用，需通过 props/emit 替代
- `$refs` 只在组件渲染完成后可用（`onMounted` 之后）
- `$refs` 不是响应式的，不应在模板中依赖其变化
- 组合式 API 中使用 `ref()` 变量名即可替代 `$refs.xxx`
- 避免在组合式 API 中使用 `$parent`，破坏了组件独立性
