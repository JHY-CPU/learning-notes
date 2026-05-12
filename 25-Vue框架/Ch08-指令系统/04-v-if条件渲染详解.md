# v-if 条件渲染详解

## 一、概念说明
`v-if` 根据表达式的真假值**条件性地渲染元素**。支持 `v-else-if` 和 `v-else` 链式使用。条件为假时，元素**不会被渲染到 DOM** 中。

## 二、具体用法

### 2.1 基本用法
```vue
<template>
  <p v-if="score >= 90">优秀</p>
  <p v-else-if="score >= 60">及格</p>
  <p v-else>不及格</p>
</template>
<script setup>
import { ref } from 'vue'
const score = ref(85)
</script>
```

### 2.2 在 template 上使用
```vue
<template>
  <!-- 不创建额外 DOM 节点 -->
  <template v-if="isLoggedIn">
    <h1>欢迎回来</h1>
    <p>用户信息</p>
  </template>
  <template v-else>
    <h1>请登录</h1>
  </template>
</template>
<script setup>
import { ref } from 'vue'
const isLoggedIn = ref(false)
</script>
```

### 2.3 v-if 管理组件生命周期
```vue
<template>
  <!-- 切换时组件被销毁和重新创建 -->
  <HeavyComponent v-if="showComponent" />
</template>
<script setup>
import { ref } from 'vue'
const showComponent = ref(true)
// showComponent 从 true 变 false 时，HeavyComponent 被销毁
// 从 false 变 true 时，重新创建
</script>
```

### 2.4 与 key 配合避免复用
```vue
<template>
  <!-- Vue 会复用元素，加 key 避免 -->
  <template v-if="loginType === 'username'">
    <label>用户名</label>
    <input key="username-input" placeholder="输入用户名" />
  </template>
  <template v-else>
    <label>邮箱</label>
    <input key="email-input" placeholder="输入邮箱" />
  </template>
</template>
```

## 三、注意事项与常见陷阱
- v-if 是**真正的条件渲染**，惰性的：初始条件为假时不做任何事
- v-if 有更高的切换开销（销毁/重建组件）
- 频繁切换用 v-show，很少改变用 v-if
- v-if 和 v-for 不要在同一元素上使用（v-for 优先级更高）

## 四、v-if 的性能影响

### 4.1 切换成本分析
```vue
<script setup>
import { ref } from 'vue'

const showHeavy = ref(true)

// 每次切换 v-if，HeavyComponent 会被完整销毁和重建
// 包括：组件实例、DOM 节点、子组件、状态
// 如果组件有大量初始化逻辑（如图表），切换成本很高

// ❌ 频繁切换时用 v-if
// <HeavyChart v-if="showHeavy" />

// ✅ 频繁切换时用 v-show
// <HeavyChart v-show="showHeavy" />
</script>
```

### 4.2 与 v-show 的选择指南
```
使用 v-if：
  - 条件很少变化
  - 初始条件为 false 时不想渲染（节省初始开销）
  - 需要触发组件的创建/销毁生命周期
  - 需要配合 <template> 分组

使用 v-show：
  - 频繁切换
  - 组件初始化开销大
  - 不需要触发组件销毁
```

## 五、常见模式

### 5.1 权限控制
```vue
<template>
  <div>
    <button v-if="isAdmin">管理面板</button>
    <button v-if="isAdmin || isEditor">编辑内容</button>
    <p v-if="!isLoggedIn">请先登录</p>
    <template v-else>
      <UserProfile />
      <Dashboard />
    </template>
  </div>
</template>
```

### 5.2 加载状态
```vue
<template>
  <div v-if="loading" class="skeleton">加载中...</div>
  <div v-else-if="error" class="error">{{ error.message }}</div>
  <div v-else-if="data.length === 0" class="empty">暂无数据</div>
  <DataList v-else :items="data" />
</template>
```

### 5.3 v-if + v-for 的正确用法
```vue
<template>
  <!-- ❌ 错误：v-if 和 v-for 在同一元素 -->
  <li v-for="item in items" v-if="item.active" :key="item.id">
    {{ item.name }}
  </li>

  <!-- ✅ 正确：用 computed 过滤 -->
  <li v-for="item in activeItems" :key="item.id">
    {{ item.name }}
  </li>

  <!-- ✅ 正确：用 <template> 包裹 -->
  <template v-for="item in items" :key="item.id">
    <li v-if="item.active">{{ item.name }}</li>
  </template>
</template>
<script setup>
import { computed } from 'vue'
const activeItems = computed(() => items.value.filter(i => i.active))
</script>
```
