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
