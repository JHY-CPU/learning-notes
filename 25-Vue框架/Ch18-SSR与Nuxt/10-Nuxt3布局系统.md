# Nuxt3布局系统

## 一、概念说明

Nuxt 3 的布局系统通过 `layouts/` 目录管理页面的整体结构框架。布局组件包含导航栏、侧边栏、页脚等公共部分，页面只需关注自身内容。使用 `<NuxtLayout>` 组件包裹页面内容来应用布局。

## 二、具体用法

### 默认布局

```vue
<!-- layouts/default.vue -->
<template>
  <div class="layout">
    <header class="nav">
      <nav>
        <NuxtLink to="/">首页</NuxtLink>
        <NuxtLink to="/about">关于</NuxtLink>
        <NuxtLink to="/blog">博客</NuxtLink>
      </nav>
    </header>

    <main class="content">
      <slot />  <!-- 页面内容渲染在这里 -->
    </main>

    <footer class="footer">
      <p>&copy; 2024 我的网站</p>
    </footer>
  </div>
</template>

<style scoped>
.layout { max-width: 1200px; margin: 0 auto; }
.nav { padding: 1rem; background: #333; color: white; }
.content { min-height: 60vh; padding: 2rem; }
.footer { padding: 1rem; text-align: center; }
</style>
```

### 自定义布局

```vue
<!-- layouts/admin.vue -->
<template>
  <div class="admin-layout">
    <aside class="sidebar">
      <h3>管理后台</h3>
      <ul>
        <li><NuxtLink to="/admin">概览</NuxtLink></li>
        <li><NuxtLink to="/admin/users">用户管理</NuxtLink></li>
        <li><NuxtLink to="/admin/posts">文章管理</NuxtLink></li>
      </ul>
    </aside>
    <section class="admin-main">
      <slot />
    </section>
  </div>
</template>

<style scoped>
.admin-layout { display: flex; }
.sidebar { width: 200px; background: #f5f5f5; padding: 1rem; }
.admin-main { flex: 1; padding: 2rem; }
</style>
```

### 页面中使用布局

```vue
<!-- pages/admin/index.vue -->
<script setup>
definePageMeta({
  layout: 'admin'  // 使用 admin 布局
})
</script>

<template>
  <div>
    <h1>管理后台概览</h1>
    <p>当前在线用户: 42</p>
  </div>
</template>
```

```vue
<!-- pages/index.vue -->
<script setup>
// 不指定 layout，默认使用 layouts/default.vue
</script>

<template>
  <div>
    <h1>欢迎来到首页</h1>
  </div>
</template>
```

### 动态切换布局

```vue
<script setup>
const layout = ref('default')

function switchLayout() {
  layout.value = layout.value === 'default' ? 'admin' : 'default'
}
</script>

<template>
  <NuxtLayout :name="layout">
    <NuxtPage />
  </NuxtLayout>
  <button @click="switchLayout">切换布局</button>
</template>
```

## 三、注意事项与常见陷阱

1. **default.vue 是默认布局**：所有页面自动使用，除非指定其他布局
2. **布局中必须有 `<slot />`**：没有插槽页面内容不会渲染
3. **app.vue 中必须包含 `<NuxtLayout>`**：否则布局系统不生效
4. **布局不能嵌套**：一个页面只能使用一个布局，需要嵌套请在布局内手动包含
5. **布局文件名即布局名**：`layouts/admin.vue` 对应布局名 `'admin'`
