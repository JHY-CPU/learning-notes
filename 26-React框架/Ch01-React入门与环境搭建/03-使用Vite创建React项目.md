# 使用 Vite 创建 React 项目

## 创建项目

### 基本命令

```bash
# 使用 npm
npm create vite@latest my-react-app -- --template react

# 使用 pnpm
pnpm create vite@latest my-react-app --template react

# 使用 yarn
yarn create vite my-react-app --template react
```

Vite 提供了多种模板选择：

| 模板名 | 说明 |
|--------|------|
| `react` | React + JavaScript |
| `react-ts` | React + TypeScript |
| `react-swc` | React + SWC（更快的编译器） |
| `react-swc-ts` | React + SWC + TypeScript |

> **推荐**：初学者用 `react`，有经验的开发者用 `react-swc-ts`。

### 启动开发服务器

```bash
cd my-react-app
npm install          # 安装依赖
npm run dev          # 启动开发服务器
```

终端会输出：

```
  VITE v5.x.x  ready in 300 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

打开浏览器访问 `http://localhost:5173/` 即可看到 React 默认页面。

## 项目目录结构

使用 `react` 模板创建的项目结构如下：

```
my-react-app/
├── public/              # 静态资源目录（不会被构建处理）
│   └── vite.svg         # Vite logo
├── src/                 # 源代码目录
│   ├── assets/          # 需要构建处理的资源
│   │   └── react.svg
│   ├── App.css          # App 组件样式
│   ├── App.jsx          # 根组件
│   ├── index.css        # 全局样式
│   └── main.jsx         # 应用入口
├── .gitignore           # Git 忽略文件
├── eslint.config.js     # ESLint 配置
├── index.html           # HTML 入口（Vite 的特殊之处）
├── package.json         # 项目依赖和脚本
├── vite.config.js       # Vite 配置文件
└── README.md
```

### 核心文件说明

**index.html** - 入口 HTML 文件

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Vite + React</title>
  </head>
  <body>
    <div id="root"></div>
    <!-- 注意：Vite 的 index.html 在根目录，不在 public/ 中 -->
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```

> 与 CRA 不同，Vite 的 `index.html` 在项目根目录而非 `public/` 目录中。

**main.jsx** - 应用入口

```jsx
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'
import './index.css'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
```

**App.jsx** - 根组件

```jsx
import { useState } from 'react'
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  return (
    <div>
      <h1>Hello React + Vite</h1>
      <button onClick={() => setCount(count + 1)}>
        点击次数：{count}
      </button>
    </div>
  )
}

export default App
```

## npm 脚本详解

`package.json` 中的 `scripts` 字段定义了可用的命令：

```json
{
  "scripts": {
    "dev": "vite",              // 启动开发服务器
    "build": "vite build",      // 生产构建
    "preview": "vite preview",  // 预览构建结果
    "lint": "eslint ."          // 代码检查
  }
}
```

```bash
npm run dev       # 启动开发服务器（端口 5173）
npm run build     # 构建生产版本（输出到 dist/）
npm run preview   # 在本地预览生产构建
npm run lint      # 运行 ESLint 检查
```

## 环境变量

Vite 使用 `.env` 文件管理环境变量，变量必须以 `VITE_` 前缀开头才能在客户端代码中访问。

### .env 文件

```bash
# .env - 所有环境共享
VITE_APP_TITLE=我的React应用
VITE_API_BASE_URL=http://localhost:3000/api
```

```bash
# .env.development - 开发环境
VITE_API_BASE_URL=http://localhost:3000/api
VITE_DEBUG=true
```

```bash
# .env.production - 生产环境
VITE_API_BASE_URL=https://api.example.com
VITE_DEBUG=false
```

### 在代码中使用

```jsx
// 通过 import.meta.env 访问
console.log(import.meta.env.VITE_APP_TITLE)    // "我的React应用"
console.log(import.meta.env.VITE_API_BASE_URL) // "http://localhost:3000/api"
console.log(import.meta.env.MODE)              // "development" | "production"
console.log(import.meta.env.DEV)               // true | false
console.log(import.meta.env.PROD)              // true | false
```

> **重要**：没有 `VITE_` 前缀的变量不会暴露给客户端代码，适合存放敏感信息（如数据库密码）。

## Vite 配置

`vite.config.js` 是 Vite 的配置文件：

```js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],

  // 开发服务器配置
  server: {
    port: 3000,           // 修改端口
    open: true,           // 自动打开浏览器
    proxy: {              // 代理配置（解决跨域）
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },

  // 构建配置
  build: {
    outDir: 'dist',       // 输出目录
    sourcemap: false,     // 是否生成 source map
    minify: 'esbuild',    // 压缩方式
  },

  // 路径别名
  resolve: {
    alias: {
      '@': '/src',        // 用 @ 代替 /src 路径
    },
  },
})
```

### 路径别名的使用

配置别名后，可以简化导入路径：

```jsx
// 不使用别名
import Button from '../../components/Button'
import { fetchData } from '../../utils/api'

// 使用 @ 别名
import Button from '@/components/Button'
import { fetchData } from '@/utils/api'
```

> 使用别名时，如果项目使用了 TypeScript，还需要在 `jsconfig.json` 或 `tsconfig.json` 中配置对应的路径映射。

### jsconfig.json（路径提示）

```json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"]
    }
  }
}
```

## 常见操作

### 安装常用依赖

```bash
# 路由
pnpm add react-router-dom

# HTTP 请求
pnpm add axios

# 状态管理
pnpm add zustand

# UI 组件库
pnpm add antd

# CSS 工具
pnpm add sass
```

### 修改端口

在 `vite.config.js` 中：

```js
server: {
  port: 3000,
}
```

或在命令行中：

```bash
npx vite --port 3000
```

## 小结

Vite 作为现代前端构建工具，具有以下优势：

- **极速启动**：利用浏览器原生 ES Module，无需打包即可运行
- **即时热更新**：HMR 速度不受项目大小影响
- **开箱即用**：支持 TypeScript、JSX、CSS 等
- **配置简洁**：相比 Webpack 配置量大幅减少
- **生态丰富**：插件系统完善，兼容 Rollup 插件

推荐所有新项目使用 Vite 作为构建工具。
