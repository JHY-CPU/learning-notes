# React 项目结构详解

## 标准项目结构

一个组织良好的 React 项目结构能让代码易于维护和扩展。以下是各规模项目的推荐目录组织方式。

## 文件详解

### index.html

项目入口 HTML 文件，位于根目录（Vite 特点）。React 应用最终会挂载到 `<div id="root">` 中。

```html
<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" href="/favicon.ico" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>React App</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```

### main.jsx

应用入口文件，负责将 React 组件渲染到 DOM 中：

```jsx
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'
import './styles/global.css'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
```

### App.jsx

根组件，通常是整个应用的顶层组件：

```jsx
import { BrowserRouter } from 'react-router-dom'
import Header from './components/Header'
import Footer from './components/Footer'
import Router from './routes'

function App() {
  return (
    <BrowserRouter>
      <Header />
      <main>
        <Router />
      </main>
      <Footer />
    </BrowserRouter>
  )
}

export default App
```

## 目录结构详解

### src/components/ - 通用组件

存放可复用的 UI 组件，不包含业务逻辑：

```
components/
├── Button/
│   ├── Button.jsx          # 组件实现
│   ├── Button.module.css   # 组件样式
│   └── index.js            # 导出入口
├── Modal/
│   ├── Modal.jsx
│   ├── Modal.module.css
│   └── index.js
└── Card/
    ├── Card.jsx
    ├── Card.module.css
    └── index.js
```

`index.js` 的作用是简化导入路径：

```jsx
// 有 index.js
import Button from './components/Button'

// 没有 index.js，需要写完整路径
import Button from './components/Button/Button'
```

```jsx
// Button/index.js
export { default } from './Button'
```

### src/pages/ - 页面组件

存放各页面的组件，通常与路由对应：

```
pages/
├── Home/
│   ├── Home.jsx
│   ├── Home.module.css
│   └── index.js
├── About/
│   ├── About.jsx
│   └── index.js
├── User/
│   ├── User.jsx
│   ├── UserProfile.jsx      # 页面内的子组件
│   ├── User.module.css
│   └── index.js
└── NotFound/
    ├── NotFound.jsx
    └── index.js
```

### src/hooks/ - 自定义 Hooks

存放自定义 React Hooks，用于复用状态逻辑：

```
hooks/
├── useAuth.js          # 认证相关
├── useFetch.js         # 数据请求
├── useDebounce.js      # 防抖
├── useLocalStorage.js  # 本地存储
└── useMediaQuery.js    # 媒体查询
```

```jsx
// hooks/useAuth.js
import { useState, useEffect } from 'react'

export function useAuth() {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // 检查登录状态
    const token = localStorage.getItem('token')
    if (token) {
      // 验证 token...
      setUser({ name: '用户' })
    }
    setLoading(false)
  }, [])

  return { user, loading }
}
```

### src/utils/ - 工具函数

存放纯函数工具，不依赖 React：

```
utils/
├── format.js       # 格式化工具
├── validate.js     # 验证工具
├── constants.js    # 常量定义
└── helpers.js      # 通用辅助函数
```

```jsx
// utils/format.js
export const formatDate = (date) => {
  return new Intl.DateTimeFormat('zh-CN').format(new Date(date))
}

export const formatCurrency = (amount) => {
  return new Intl.NumberFormat('zh-CN', {
    style: 'currency',
    currency: 'CNY',
  }).format(amount)
}
```

### src/services/ - API 服务

存放与后端交互的接口封装：

```
services/
├── api.js           # axios 实例配置
├── userApi.js       # 用户相关接口
├── productApi.js    # 产品相关接口
└── authApi.js       # 认证相关接口
```

```jsx
// services/api.js
import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL,
  timeout: 10000,
})

// 请求拦截器
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// 响应拦截器
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  },
)

export default api
```

### src/routes/ - 路由配置

存放路由定义：

```
routes/
├── index.jsx         # 路由配置
├── PrivateRoute.jsx  # 路由守卫
└── routeConfig.js    # 路由表
```

```jsx
// routes/index.jsx
import { Routes, Route } from 'react-router-dom'
import Home from '@/pages/Home'
import About from '@/pages/About'
import User from '@/pages/User'
import NotFound from '@/pages/NotFound'

export default function Router() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/about" element={<About />} />
      <Route path="/user/:id" element={<User />} />
      <Route path="*" element={<NotFound />} />
    </Routes>
  )
}
```

### src/styles/ - 全局样式

```
styles/
├── global.css        # 全局基础样式
├── variables.css     # CSS 变量
├── mixins.css        # CSS 复用
└── reset.css         # 浏览器重置
```

```css
/* styles/variables.css */
:root {
  --color-primary: #1890ff;
  --color-success: #52c41a;
  --color-warning: #faad14;
  --color-error: #ff4d4f;
  --font-size-base: 14px;
  --border-radius: 4px;
}
```

### src/assets/ - 静态资源

存放需要经过构建处理的资源文件（图片、字体、SVG 等）：

```
assets/
├── images/
│   ├── logo.png
│   └── background.jpg
├── icons/
│   └── arrow.svg
└── fonts/
    └── custom-font.woff2
```

### public/ - 公共静态资源

存放不需要构建处理的文件，直接复制到输出目录：

```
public/
├── favicon.ico
├── robots.txt
└── manifest.json
```

## 不同规模的项目组织

### 小型项目（5-10 个组件）

```
src/
├── components/        # 所有组件放一起
│   ├── Header.jsx
│   ├── Footer.jsx
│   └── Card.jsx
├── App.jsx
├── main.jsx
└── index.css
```

### 中型项目（10-50 个组件）

```
src/
├── components/        # 通用组件
│   ├── Button/
│   ├── Modal/
│   └── Form/
├── pages/             # 页面组件
│   ├── Home/
│   ├── About/
│   └── User/
├── hooks/             # 自定义 Hooks
├── utils/             # 工具函数
├── services/          # API 服务
├── routes/            # 路由
├── styles/            # 样式
└── App.jsx
```

### 大型项目（50+ 组件，多人协作）

```
src/
├── features/          # 按功能模块组织
│   ├── auth/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── services/
│   │   └── pages/
│   ├── dashboard/
│   │   ├── components/
│   │   ├── hooks/
│   │   └── pages/
│   └── settings/
│       ├── components/
│       └── pages/
├── shared/            # 共享资源
│   ├── components/    # 通用 UI 组件
│   ├── hooks/         # 通用 Hooks
│   ├── utils/         # 工具函数
│   └── styles/        # 全局样式
├── routes/            # 路由配置
├── App.jsx
└── main.jsx
```

## 最佳实践总结

1. **按功能而非类型组织**：大型项目推荐 `features/` 目录结构
2. **组件一个文件夹**：每个组件（及其样式、测试）放在独立文件夹中
3. **统一导出入口**：每个文件夹使用 `index.js` 做统一导出
4. **使用路径别名**：配置 `@/` 别名简化导入路径
5. **保持目录一致**：团队内保持统一的目录规范
6. **避免过深嵌套**：目录层级一般不超过 4 层
