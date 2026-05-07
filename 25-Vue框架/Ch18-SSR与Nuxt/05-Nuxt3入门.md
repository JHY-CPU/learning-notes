# Nuxt3入门

## 一、概念说明

Nuxt 是基于 Vue 的全栈框架，内置 SSR、文件路由、自动导入等特性，开箱即用。Nuxt 3 基于 Vue 3 + Vite + Nitro 构建，支持 SSR、SSG、ISR 等多种渲染模式，是 Vue 生态中最成熟的 SSR 解决方案。

## 二、具体用法

### 创建项目

```bash
# 使用 nuxi CLI 创建 Nuxt 3 项目
npx nuxi@latest init my-nuxt-app
cd my-nuxt-app
npm install
npm run dev
# 终端输出：Nuxt Dev Server running at http://localhost:3000
```

### 项目目录结构

```
my-nuxt-app/
├── app.vue              # 根组件（必须存在）
├── pages/               # 页面目录（文件路由）
│   ├── index.vue        # → /
│   └── about.vue        # → /about
├── components/          # 组件自动导入
├── composables/         # 组合式函数自动导入
├── layouts/             # 布局模板
├── middleware/          # 路由中间件
├── plugins/             # 插件
├── server/              # 服务端API
├── public/              # 静态资源
├── assets/              # 样式、字体等
├── nuxt.config.ts       # Nuxt 配置
└── package.json
```

### 根组件与页面

```vue
<!-- app.vue - 最简根组件 -->
<template>
  <div>
    <NuxtPage />
  </div>
</template>
```

```vue
<!-- pages/index.vue -->
<script setup>
// 无需手动导入 ref、computed，Nuxt 自动导入
const title = ref('欢迎使用 Nuxt 3')
const features = reactive([
  '服务端渲染 SSR',
  '自动导入组件',
  '文件系统路由',
  'TypeScript 支持'
])
</script>

<template>
  <div>
    <h1>{{ title }}</h1>
    <ul>
      <li v-for="(item, i) in features" :key="i">{{ item }}</li>
    </ul>
  </div>
</template>

<!-- 浏览器访问 http://localhost:3000 显示完整页面 -->
<!-- 右键查看源代码可看到服务端渲染的HTML内容 -->
```

### Nuxt 配置文件

```ts
// nuxt.config.ts
export default defineNuxtConfig({
  devtools: { enabled: true },

  // 应用头部配置
  app: {
    head: {
      title: '我的 Nuxt 应用',
      meta: [
        { charset: 'utf-8' },
        { name: 'viewport', content: 'width=device-width, initial-scale=1' }
      ]
    }
  },

  // 模块扩展
  modules: [
    '@pinia/nuxt',
    '@nuxtjs/tailwindcss'
  ]
})
```

## 三、注意事项与常见陷阱

1. **app.vue 是必须的**：缺少根组件 Nuxt 无法启动，即使你只用 pages 目录
2. **自动导入是默认行为**：`ref`、`computed`、`watch` 等无需手动 import，但需注意可读性
3. **不要在 nuxt.config 中写运行时逻辑**：配置文件在构建时执行，无法访问浏览器 API
4. **pages 目录是可选的**：不创建 pages 目录时，所有内容写在 app.vue 中
5. **服务端和客户端区分**：使用 `process.server` 和 `process.client` 判断运行环境
