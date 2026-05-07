# 条件渲染 v-if

## 一、概念说明

`v-if` 指令用于条件性地渲染元素。当表达式为真值时渲染元素，为假值时从 DOM 中移除。支持 `v-else-if` 和 `v-else` 链式使用。

```vue
<script setup>
import { ref } from 'vue'

const score = ref(85)
const isLoggedIn = ref(true)
</script>

<template>
  <!-- 基本用法 -->
  <p v-if="isLoggedIn">欢迎回来！</p>
  <p v-else>请先登录</p>

  <!-- 多条件分支 -->
  <p v-if="score >= 90">优秀</p>
  <p v-else-if="score >= 60">及格</p>
  <p v-else>不及格</p>
</template>
```

## 二、具体用法

### 2.1 在 `<template>` 上使用 v-if

```vue
<template>
  <!-- 一次性切换多个元素，不产生额外 DOM 节点 -->
  <template v-if="isLoggedIn">
    <h1>欢迎回来</h1>
    <nav>导航栏内容</nav>
    <p>用户信息</p>
  </template>
  <template v-else>
    <h1>请登录</h1>
    <p>登录表单</p>
  </template>
</template>
```

### 2.2 v-if 控制组件渲染

```vue
<script setup>
import { ref } from 'vue'
import AdminPanel from './AdminPanel.vue'
import UserPanel from './UserPanel.vue'

const role = ref('admin')
</script>

<template>
  <AdminPanel v-if="role === 'admin'" />
  <UserPanel v-else />
</template>
```

### 2.3 与 key 配合复用问题

```vue
<template>
  <!-- Vue 会复用元素（相同的 key），用 key 避免 -->
  <template v-if="loginType === 'username'">
    <label>用户名</label>
    <input placeholder="输入用户名" key="username-input" />
  </template>
  <template v-else>
    <label>邮箱</label>
    <input placeholder="输入邮箱" key="email-input" />
  </template>
</template>
```

## 三、注意事项与常见陷阱

- `v-if` 是"真正"的条件渲染，会销毁和重建组件（触发生命周期）
- `v-if` 有更高的切换开销（涉及 DOM 增删）
- `v-else` 和 `v-else-if` 必须紧跟在 `v-if` 或 `v-else-if` 后面
- 不推荐 `v-if` 和 `v-for` 同时用在同一个元素上（Vue 3 中 v-for 优先级更高）
- 频繁切换的场景用 `v-show` 更高效
