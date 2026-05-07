# React Router 基础

React Router 是 React 生态中最流行的路由库，用于构建单页面应用（SPA）中的页面导航。

---

## 一、安装与基本配置

### 1.1 安装

```bash
npm install react-router-dom
```

### 1.2 基础配置

```jsx
// main.jsx
import { BrowserRouter } from 'react-router-dom';
import App from './App';

ReactDOM.createRoot(document.getElementById('root')).render(
  <BrowserRouter>
    <App />
  </BrowserRouter>
);
```

### 1.3 最简路由

```jsx
import { Routes, Route } from 'react-router-dom';

function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/about" element={<About />} />
      <Route path="/contact" element={<Contact />} />
    </Routes>
  );
}
```

---

## 二、核心组件

### 2.1 BrowserRouter vs HashRouter vs MemoryRouter

```jsx
// BrowserRouter（最常用）：使用 HTML5 History API
// URL 格式：https://example.com/about
import { BrowserRouter } from 'react-router-dom';

// HashRouter：使用 URL hash
// URL 格式：https://example.com/#/about
// 适用：不支持 History API 的环境、静态文件服务器
import { HashRouter } from 'react-router-dom';

// MemoryRouter：不修改 URL，将路由记录保存在内存中
// 适用：测试、React Native、非浏览器环境
import { MemoryRouter } from 'react-router-dom';
```

**选择建议**：

| 路由器 | 适用场景 |
|---|---|
| BrowserRouter | 大多数 Web 应用 |
| HashRouter | 静态托管（GitHub Pages）、旧服务器 |
| MemoryRouter | 测试、组件库文档、React Native |

### 2.2 Routes 与 Route

```jsx
import { Routes, Route } from 'react-router-dom';

function App() {
  return (
    <Routes>
      {/* 基础路由 */}
      <Route path="/" element={<Home />} />

      {/* 动态参数 */}
      <Route path="/users/:id" element={<UserDetail />} />

      {/* 可选参数 */}
      <Route path="/posts/:id?" element={<Posts />} />

      {/* 通配符（404 页面） */}
      <Route path="*" element={<NotFound />} />
    </Routes>
  );
}
```

### 2.3 Link 组件

```jsx
import { Link } from 'react-router-dom';

function Navigation() {
  return (
    <nav>
      <Link to="/">首页</Link>
      <Link to="/about">关于</Link>
      <Link to="/users">用户列表</Link>

      {/* 带参数的链接 */}
      <Link to="/users/123">用户 123</Link>

      {/* 带查询参数 */}
      <Link to="/search?q=react&page=1">搜索 React</Link>

      {/* 带状态 */}
      <Link to="/dashboard" state={{ from: 'home' }}>
        Dashboard
      </Link>

      {/* 替换当前历史记录 */}
      <Link to="/login" replace>登录</Link>
    </nav>
  );
}
```

### 2.4 NavLink

`NavLink` 是 `Link` 的特殊版本，可以知道当前路由是否"活跃"。

```jsx
import { NavLink } from 'react-router-dom';

function Navigation() {
  return (
    <nav>
      {/* 使用 className 函数 */}
      <NavLink
        to="/"
        className={({ isActive }) => isActive ? 'active' : ''}
      >
        首页
      </NavLink>

      {/* 使用 style 函数 */}
      <NavLink
        to="/about"
        style={({ isActive }) => ({
          color: isActive ? 'red' : 'blue',
          fontWeight: isActive ? 'bold' : 'normal',
        })}
      >
        关于
      </NavLink>

      {/* end 属性：精确匹配 */}
      <NavLink to="/" end>
        首页（只有 / 时才 active，/about 不会激活）
      </NavLink>
    </nav>
  );
}
```

---

## 三、动态路由参数

### 3.1 路径参数 (URL Params)

```jsx
// 路由定义
<Route path="/users/:userId" element={<UserDetail />} />
<Route path="/posts/:postId/comments/:commentId" element={<Comment />} />

// 获取参数
import { useParams } from 'react-router-dom';

function UserDetail() {
  const { userId } = useParams();
  // URL: /users/123 → userId = "123"（始终为字符串）

  return <div>用户 ID: {userId}</div>;
}

function Comment() {
  const { postId, commentId } = useParams();
  // URL: /posts/42/comments/7 → postId = "42", commentId = "7"

  return <div>文章 {postId} 的评论 {commentId}</div>;
}
```

### 3.2 查询参数 (Query Params / Search Params)

```jsx
import { useSearchParams } from 'react-router-dom';

function SearchPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  // URL: /search?q=react&page=2

  const query = searchParams.get('q');       // "react"
  const page = searchParams.get('page');     // "2"
  const sort = searchParams.get('sort');     // null（不存在时）

  const handleSearch = (newQuery) => {
    setSearchParams({ q: newQuery, page: '1' });
  };

  return (
    <div>
      <input
        value={query || ''}
        onChange={(e) => handleSearch(e.target.value)}
      />
      <p>搜索: {query}, 第 {page || 1} 页</p>
    </div>
  );
}
```

---

## 四、索引路由与嵌套路由

### 4.1 索引路由

索引路由在父路由的路径被精确匹配时渲染。

```jsx
<Routes>
  <Route path="/" element={<Layout />}>
    {/* 索引路由：path="/" 精确匹配时显示 */}
    <Route index element={<Home />} />

    {/* 子路由 */}
    <Route path="about" element={<About />} />
    <Route path="dashboard" element={<Dashboard />} />
  </Route>
</Routes>
```

### 4.2 嵌套路由与 Outlet

```jsx
import { Outlet } from 'react-router-dom';

// 布局组件：使用 Outlet 渲染子路由
function Layout() {
  return (
    <div>
      <header>
        <nav>
          <Link to="/">首页</Link>
          <Link to="/dashboard">仪表盘</Link>
          <Link to="/settings">设置</Link>
        </nav>
      </header>
      <main>
        <Outlet />  {/* 子路由内容在这里渲染 */}
      </main>
      <footer>页脚</footer>
    </div>
  );
}

// 路由配置
<Routes>
  <Route path="/" element={<Layout />}>
    <Route index element={<Home />} />
    <Route path="dashboard" element={<Dashboard />} />
    <Route path="settings" element={<Settings />} />
  </Route>
</Routes>
```

---

## 五、404 / Not Found 路由

```jsx
function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/about" element={<About />} />
      <Route path="/products/:id" element={<Product />} />

      {/* 匹配所有未定义路径 */}
      <Route path="*" element={<NotFound />} />
    </Routes>
  );
}

function NotFound() {
  const location = useLocation();

  return (
    <div>
      <h1>404</h1>
      <p>页面 <code>{location.pathname}</code> 不存在</p>
      <Link to="/">返回首页</Link>
    </div>
  );
}
```

---

## 六、路由配置方式对比

### 6.1 声明式（推荐）

```jsx
// JSX 声明式配置
<Routes>
  <Route path="/" element={<Layout />}>
    <Route index element={<Home />} />
    <Route path="products" element={<Products />} />
    <Route path="products/:id" element={<ProductDetail />} />
  </Route>
</Routes>
```

### 6.2 对象配置（React Router 6.4+）

```jsx
import { createBrowserRouter, RouterProvider } from 'react-router-dom';

const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout />,
    children: [
      { index: true, element: <Home /> },
      { path: 'products', element: <Products /> },
      { path: 'products/:id', element: <ProductDetail /> },
      { path: '*', element: <NotFound /> },
    ],
  },
]);

function App() {
  return <RouterProvider router={router} />;
}
```

---

## 七、完整示例

```jsx
import { BrowserRouter, Routes, Route, Link, NavLink, Outlet } from 'react-router-dom';

// 布局
function RootLayout() {
  return (
    <div className="app">
      <header>
        <h1>我的应用</h1>
        <nav>
          <NavLink to="/" end className={({ isActive }) => isActive ? 'active' : ''}>
            首页
          </NavLink>
          <NavLink to="/products" className={({ isActive }) => isActive ? 'active' : ''}>
            产品
          </NavLink>
          <NavLink to="/about" className={({ isActive }) => isActive ? 'active' : ''}>
            关于
          </NavLink>
        </nav>
      </header>
      <main>
        <Outlet />
      </main>
    </div>
  );
}

// 页面组件
function Home() {
  return <h2>欢迎来到首页</h2>;
}

function Products() {
  return <h2>产品列表</h2>;
}

function About() {
  return <h2>关于我们</h2>;
}

function NotFound() {
  return (
    <div>
      <h2>404 - 页面未找到</h2>
      <Link to="/">返回首页</Link>
    </div>
  );
}

// 应用
function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<RootLayout />}>
          <Route index element={<Home />} />
          <Route path="products" element={<Products />} />
          <Route path="about" element={<About />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
```
