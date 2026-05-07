# Nuxt3部署

## 一、概念说明

Nuxt 3 支持多种部署方式：Node.js 服务器（SSR）、静态站点（SSG）、以及各大云平台（Vercel、Netlify）。通过 `nuxt.config.ts` 的 `routeRules` 可混合使用多种渲染模式。Nitro 引擎自动处理部署适配。

## 二、具体用法

### Node.js 服务器部署

```bash
# 构建 SSR 应用
npm run build
# 生成 .output/ 目录

# 预览构建结果
node .output/server/index.mjs
# 终端输出：Listening on http://localhost:3000
```

```ts
// nuxt.config.ts - SSR 模式（默认）
export default defineNuxtConfig({
  ssr: true,  // 默认开启 SSR
})
```

### 静态站点生成

```ts
// nuxt.config.ts - SSG 模式
export default defineNuxtConfig({
  ssr: true,
  nitro: {
    prerender: {
      routes: ['/', '/about', '/blog/1', '/blog/2']
    }
  }
})
```

```bash
# 生成静态文件
npx nuxi generate
# 输出到 .output/public/ 目录
# 可直接部署到 GitHub Pages、Netlify 等静态托管
```

### Vercel 部署

```ts
// nuxt.config.ts
export default defineNuxtConfig({
  // Vercel 自动检测 Nuxt，零配置部署
  routeRules: {
    '/': { prerender: true },         // 首页预渲染
    '/api/**': { cors: true },        // API 跨域
    '/admin/**': { ssr: false }       // 管理页面 CSR
  }
})
```

```bash
# 使用 Vercel CLI 部署
npm i -g vercel
vercel
# 自动构建并部署到 Vercel 平台
```

### Netlify 部署

```bash
# 安装 Netlify CLI
npm i -g netlify-cli

# 构建并部署
netlify deploy --prod --dir=.output
# 部署到 Netlify，自动配置 serverless 函数
```

### 混合渲染模式

```ts
// nuxt.config.ts
export default defineNuxtConfig({
  routeRules: {
    '/': { prerender: true },              // SSG：构建时生成
    '/products/**': { swr: 3600 },         // ISR：1小时重新验证
    '/admin/**': { ssr: false },           // CSR：纯客户端渲染
    '/api/**': { cors: true }              // API 路由
  }
})
```

## 三、注意事项与常见陷阱

1. **`ssr: false` 变成纯 SPA**：设置后所有页面都是客户端渲染，失去 SSR 优势
2. **静态部署不支持动态路由**：`[id].vue` 的页面需要在 prerender.routes 中预列出
3. **环境变量需要 NUXT_ 前缀**：生产环境的环境变量需以 `NUXT_` 开头才能在应用中访问
4. **.output 目录包含完整运行时**：Node.js 部署只需复制此目录，无需 src
5. **Vercel/Netlify 默认支持 SSR**：无需额外配置 serverless 函数
