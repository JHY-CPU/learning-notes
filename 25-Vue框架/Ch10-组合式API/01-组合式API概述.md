# 组合式API概述

## 一、概念说明

Vue 3 引入了**组合式API (Composition API)**，它是一组基于函数的API，允许我们更灵活地组织组件逻辑。相比选项式API，组合式API解决了逻辑关注点分散的问题。

```vue
<template>
  <div>
    <p>计数: {{ count }}</p>
    <p>双倍: {{ double }}</p>
    <button @click="increment">+1</button>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const count = ref(0)
const double = computed(() => count.value * 2)
const increment = () => count.value++
</script>
```

**为什么需要组合式API？**
- 选项式API中，相同逻辑的data、methods、computed分散在不同位置
- 组件变大后，逻辑关注点难以追踪
- 组合式API将同一逻辑关注点的代码集中在一起

## 二、具体用法

组合式API的核心是`setup()`函数（或`<script setup>`语法糖），在此作用域内可以使用：

- **响应式状态**：`ref()`、`reactive()`
- **计算属性**：`computed()`
- **生命周期钩子**：`onMounted()`、`onUnmounted()`等
- **侦听器**：`watch()`、`watchEffect()`

```vue
<script setup>
import { ref, reactive, computed, watch, onMounted } from 'vue'

// 响应式状态
const count = ref(0)
const user = reactive({ name: '张三', age: 20 })

// 计算属性
const info = computed(() => `${user.name} - ${user.age}岁`)

// 侦听器
watch(count, (newVal) => {
  console.log('计数变为:', newVal)
})

// 生命周期
onMounted(() => {
  console.log('组件已挂载')
})
</script>
```

## 三、注意事项与常见陷阱

1. **组合式API和选项式API可以共存**，但推荐选择一种风格保持一致
2. 在`<script setup>`中，顶层导入和变量自动暴露给模板
3. `ref`在JS中需要`.value`访问，在模板中自动解包
4. 组合式API更适合TypeScript类型推断
5. 逻辑复用通过**组合式函数 (Composables)** 实现，替代mixins

## 四、组合式 API vs 选项式 API 详细对比

```vue
<!-- 选项式 API：按类型组织 -->
<script>
export default {
  data() {
    return {
      search: '',
      users: [],
      loading: false
    }
  },
  computed: {
    filteredUsers() {
      return this.users.filter(u => u.name.includes(this.search))
    }
  },
  methods: {
    async fetchUsers() {
      this.loading = true
      this.users = await fetch('/api/users').then(r => r.json())
      this.loading = false
    }
  },
  mounted() {
    this.fetchUsers()
  }
}
</script>

<!-- 组合式 API：按逻辑关注点组织 -->
<script setup>
import { ref, computed, onMounted } from 'vue'

// 数据获取逻辑
const users = ref([])
const loading = ref(false)
async function fetchUsers() {
  loading.value = true
  users.value = await fetch('/api/users').then(r => r.json())
  loading.value = false
}
onMounted(fetchUsers)

// 搜索过滤逻辑
const search = ref('')
const filteredUsers = computed(() =>
  users.value.filter(u => u.name.includes(search.value))
)
</script>
```

## 五、何时选择组合式 API

| 场景 | 推荐 |
|------|------|
| 复杂组件，逻辑点多 | 组合式 API |
| 简单组件，少量 data/methods | 选项式 API |
| 需要 TypeScript | 组合式 API |
| 需要逻辑复用（composables） | 组合式 API |
| 团队不熟悉 Composition API | 选项式 API |
| 使用 Vue 2 迁移 | 选项式 API 过渡 |

## 六、组合式 API 的核心优势

1. **逻辑关注点集中**：相关代码放在一起，而非分散在 data/methods/computed 中
2. **更好的类型推断**：TypeScript 支持更自然
3. **灵活的逻辑复用**：Composables 替代 mixins，无命名冲突
4. **更小的打包体积**：tree-shaking 更友好，按需导入
5. **适用于非 Vue 场景**：响应式系统可独立使用（`@vue/reactivity`）
