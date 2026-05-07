# 创建Vue项目（Vite）

## 一、概念说明

**Vite** 是 Vue 团队推荐的构建工具，由尤雨溪开发。它利用浏览器原生 ES Module 支持，在开发环境下无需打包，启动极快。`create-vue` 是官方的项目脚手架，基于 Vite 快速生成 Vue 3 项目模板。

Vite 的核心优势：冷启动秒开、HMR（热模块替换）即时生效、按需编译。相比 Webpack，Vite 在开发体验上有质的提升。

## 二、具体用法

### 2.1 创建项目

```bash
# 使用 pnpm 创建（推荐）
pnpm create vue@latest

# 交互式选择：
# ✔ Project name: my-vue-app
# ✔ TypeScript: Yes
# ✔ JSX: No
# ✔ Vue Router: Yes
# ✔ Pinia: Yes
# ✔ Vitest: Yes
# ✔ ESLint: Yes
# ✔ Prettier: Yes
```

### 2.2 项目结构

```
my-vue-app/
├── public/            # 静态资源（不经过构建）
├── src/
│   ├── assets/        # 资源文件（图片、样式等）
│   ├── components/    # 公共组件
│   ├── views/         # 页面视图
│   ├── router/        # 路由配置
│   ├── stores/        # Pinia 状态管理
│   ├── App.vue        # 根组件
│   └── main.js        # 入口文件
├── index.html         # HTML 入口
├── vite.config.js     # Vite 配置
├── package.json       # 项目依赖
└── .eslintrc.cjs      # ESLint 配置
```

### 2.3 运行项目

```bash
cd my-vue-app
pnpm install       # 安装依赖
pnpm dev           # 启动开发服务器（默认 http://localhost:5173）
pnpm build         # 构建生产版本
pnpm preview       # 预览生产构建
```

## 三、注意事项与常见陷阱

- `create-vue` 与 `vue-cli` 不同，前者基于 Vite，后者基于 Webpack（已不推荐）
- Node.js 版本需 >= 16，推荐 18+
- 首次 `pnpm install` 较慢，后续有缓存会很快
- Vite 配置文件是 `vite.config.js`，不是 `vue.config.js`
- 开发服务器默认只监听 localhost，如需局域网访问用 `pnpm dev --host`
