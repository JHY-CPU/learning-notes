# Nuxt3路由

## 一、概念说明

Nuxt 3 使用**文件系统路由**，`pages/` 目录下的文件结构自动映射为路由路径，无需手动配置 Vue Router。每个 `.vue` 文件自动成为页面，目录名成为路径段。Nuxt 在底层自动生成 `vue-router` 配置。

## 二、具体用法

### 基本路由

```
pages/
├── index.vue        # → /
├── about.vue        # → /about
├── contact.vue      # → /contact
└── blog/
    ├── index.vue    # → /blog
    └── [...slug].vue # → /blog/任意路径
```

```vue
<!-- pages/about.vue -->
<script setup>
// 自动成为 /about 路由的页面组件
const teamMembers = reactive([
  { name: '张三', role: '前端工程师' },
  { name: '李四', role: '后端工程师' }
])
</script>

<template>
  <div>
    <h1>关于我们</h1>
    <div v-for="member in teamMembers" :key="member.name">
      <h3>{{ member.name }}</h3>
      <p>{{ member.role }}</p>
    </div>
  </div>
</template>
```

### 动态路由

```
pages/
├── users/
│   ├── index.vue        # → /users
│   ├── [id].vue         # → /users/123, /users/456
│   └── [id]/
│       └── profile.vue  # → /users/123/profile
└── posts/
    └── [...slug].vue    # → /posts/a, /posts/a/b, /posts/a/b/c
```

```vue
<!-- pages/users/[id].vue -->
<script setup>
// 获取动态路由参数
const route = useRoute()
const userId = route.params.id

// userId 输出：访问 /users/123 时，userId 为 '123'
const user = ref({ id: userId, name: '加载中...' })
</script>

<template>
  <div>
    <h1>用户详情</h1>
    <p>用户ID: {{ $route.params.id }}</p>
    <p>用户名: {{ user.name }}</p>
    <NuxtLink to="/users">返回用户列表</NuxtLink>
  </div>
</template>
```

### 编程式导航

```vue
<script setup>
const router = useRouter()

function goToUser(id) {
  router.push(`/users/${id}`)
  // 跳转到 /users/123
}

function goBack() {
  router.back()
}
</script>

<template>
  <div>
    <button @click="goToUser(123)">查看用户123</button>
    <button @click="goBack">返回</button>

    <!-- NuxtLink 声明式导航 -->
    <NuxtLink to="/about" class="link">关于页面</NuxtLink>
  </div>
</template>
```

### 路由中间件

```vue
<!-- pages/dashboard.vue -->
<script setup>
definePageMeta({
  middleware: 'auth'  // 使用 auth 中间件
})
</script>

<template>
  <div>仪表盘页面（需登录）</div>
</template>
```

## 三、注意事项与常见陷阱

1. **文件名即路由**：文件名中的大写字母会自动转为小写路径，`UserProfile.vue` 对应 `/userprofile`
2. **动态参数用方括号**：`[id].vue` 是动态路由，不是 `:id.vue`
3. **通配符用 `[...slug].vue`**：捕获所有子路径，对应 Vue Router 的 `/:pathMatch(.*)*`
4. **NuxtLink 优于 router.push**：模板中使用 `NuxtLink` 享受预加载和性能优化
5. **`definePageMeta` 只能在页面组件中使用**：普通组件中使用不会生效
