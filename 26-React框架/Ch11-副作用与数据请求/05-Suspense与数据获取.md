# Ch11-05 Suspense 与数据获取

## 目录

1. [Suspense 基础概念](#1-suspense-基础概念)
2. [React.lazy 与 Suspense 代码分割](#2-reactlazy-与-suspense-代码分割)
3. [抛出 Promise 模式](#3-抛出-promise-模式)
4. [并发特性 (Concurrent Features)](#4-并发特性-concurrent-features)
5. [use() Hook (React 19)](#5-use-hook-react-19)
6. [Suspense 边界策略](#6-suspense-边界策略)
7. [Fallback UI 设计](#7-fallback-ui-设计)
8. [流式 SSR (Streaming SSR)](#8-流式-ssr-streaming-ssr)

---

## 1. Suspense 基础概念

### 1.1 什么是 Suspense

Suspense 是 React 提出的一种声明式的"等待异步操作"机制。它让组件可以在渲染时"暂停"，等待某些异步数据或代码加载完成后再继续渲染。

```jsx
import { Suspense } from "react";

function App() {
  return (
    <Suspense fallback={<div>加载中...</div>}>
      <UserProfile />
    </Suspense>
  );
}
```

### 1.2 核心思想

```
传统方式：
  组件渲染 → 检查 loading 状态 → 显示 loading UI → 数据到达 → 重新渲染显示数据

Suspense 方式：
  组件"告诉" React "我还没准备好" → React 暂停渲染 → 显示 fallback
  → 数据到达 → React 重新尝试渲染 → 显示实际内容
```

### 1.3 Suspense 的工作原理

```
1. React 渲染组件树
2. 某个组件抛出一个 Promise（表示"我还在加载"）
3. React 捕获这个 Promise
4. React 沿树向上查找最近的 <Suspense> 边界
5. 显示该边界的 fallback
6. Promise 解析后，React 重新渲染该组件
7. 渲染成功，替换 fallback 显示实际内容
```

---

## 2. React.lazy 与 Suspense 代码分割

### 2.1 React.lazy 基础

`React.lazy` 允许你延迟加载组件的代码，直到该组件首次被渲染：

```jsx
import { lazy, Suspense } from "react";

// 懒加载组件 - 只有在首次渲染时才加载代码
const Dashboard = lazy(() => import("./pages/Dashboard"));
const Settings = lazy(() => import("./pages/Settings"));
const Profile = lazy(() => import("./pages/Profile"));

function App() {
  return (
    <Suspense fallback={<div>页面加载中...</div>}>
      <Router>
        <Routes>
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/profile" element={<Profile />} />
        </Routes>
      </Router>
    </Suspense>
  );
}
```

### 2.2 命名导出的懒加载

```jsx
// 如果组件使用命名导出而非默认导出
const Dashboard = lazy(() =>
  import("./pages/Dashboard").then((module) => ({
    default: module.Dashboard,
  }))
);

// 或使用辅助函数
function namedLazy(loadModule, name) {
  return lazy(() =>
    loadModule().then((module) => ({ default: module[name] }))
  );
}

const Dashboard = namedLazy(() => import("./pages/Dashboard"), "Dashboard");
```

### 2.3 带加载超时的懒加载

```jsx
const Dashboard = lazy(() =>
  Promise.race([
    import("./pages/Dashboard"),
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error("加载超时")), 10000)
    ),
  ])
);
```

---

## 3. 抛出 Promise 模式

### 3.1 原理

Suspense 的核心机制是：组件可以在渲染过程中"抛出"一个 Promise，React 会捕获它并显示 fallback。

```js
// 简化的 Suspense 数据源实现
function wrapPromise(promise) {
  let status = "pending";
  let result;

  const suspender = promise.then(
    (res) => {
      status = "success";
      result = res;
    },
    (err) => {
      status = "error";
      result = err;
    }
  );

  return {
    read() {
      if (status === "pending") throw suspender;    // 抛出 Promise → 暂停
      if (status === "error") throw result;         // 抛出错误 → ErrorBoundary
      if (status === "success") return result;      // 返回数据
    },
  };
}

// 创建 Suspense 数据源
const userResource = wrapPromise(
  fetch("/api/user").then((res) => res.json())
);

// 在组件中使用
function UserProfile() {
  // 如果数据还没到，会抛出 Promise，触发 Suspense fallback
  const user = userResource.read();

  return <div>{user.name}</div>;
}
```

### 3.2 完整示例

```jsx
import { Suspense } from "react";

// 数据获取层
function createFetchResource(fetchFn) {
  let status = "pending";
  let result;
  let suspender;

  return {
    read() {
      if (status === "success") return result;
      if (status === "error") throw result;
      throw suspender;
    },
    fetch(...args) {
      suspender = fetchFn(...args).then(
        (res) => { status = "success"; result = res; },
        (err) => { status = "error"; result = err; }
      );
      return this;
    },
  };
}

// 使用
const userResource = createFetchResource((id) =>
  fetch(`/api/users/${id}`).then((r) => r.json())
).fetch(1);

function UserProfile() {
  const user = userResource.read();
  return <h1>{user.name}</h1>;
}

function App() {
  return (
    <Suspense fallback={<div>加载用户信息...</div>}>
      <UserProfile />
    </Suspense>
  );
}
```

---

## 4. 并发特性 (Concurrent Features)

### 4.1 Suspense 与并发模式

React 18 引入的并发特性使 Suspense 更加强大：

```jsx
import { Suspense, useTransition } from "react";

function App() {
  const [userId, setUserId] = useState(1);
  const [isPending, startTransition] = useTransition();

  const handleClick = (newId) => {
    // 使用 startTransition 标记低优先级更新
    startTransition(() => {
      setUserId(newId);
    });
  };

  return (
    <div>
      <nav>
        {[1, 2, 3].map((id) => (
          <button key={id} onClick={() => handleClick(id)}>
            用户 {id}
          </button>
        ))}
      </nav>

      {/* isPending 表示正在加载新数据，可以保持当前 UI 不变 */}
      <div style={{ opacity: isPending ? 0.7 : 1 }}>
        <Suspense fallback={<div>加载中...</div>}>
          <UserProfile userId={userId} />
        </Suspense>
      </div>
    </div>
  );
}
```

### 4.2 Suspense 的 useTransition

```jsx
function App() {
  const [tab, setTab] = useState("posts");
  const [isPending, startTransition] = useTransition();

  function handleTabChange(newTab) {
    startTransition(() => {
      setTab(newTab);
    });
  }

  return (
    <div>
      <TabBar activeTab={tab} onChange={handleTabChange} />
      {isPending && <div>加载中...</div>}
      <Suspense fallback={<TabSkeleton />}>
        {tab === "posts" && <Posts />}
        {tab === "comments" && <Comments />}
        {tab === "albums" && <Albums />}
      </Suspense>
    </div>
  );
}
```

---

## 5. use() Hook (React 19)

### 5.1 基本用法

React 19 引入了 `use()` Hook，可以在组件中"读取" Promise 或 Context 的值：

```jsx
import { use, Suspense } from "react";

// 异步数据获取函数
async function fetchUser(id) {
  const res = await fetch(`/api/users/${id}`);
  return res.json();
}

function UserProfile({ userPromise }) {
  // use() 会在 Promise 解析前"暂停"组件渲染
  const user = use(userPromise);

  return <div>{user.name}</div>;
}

function App() {
  // 创建 Promise（不是在 useEffect 中，而是在渲染时）
  const userPromise = fetchUser(1);

  return (
    <Suspense fallback={<div>加载中...</div>}>
      <UserProfile userPromise={userPromise} />
    </Suspense>
  );
}
```

### 5.2 use() 与 Context

```jsx
import { use, createContext, Suspense } from "react";

const ThemeContext = createContext(null);

function ThemedButton() {
  // use() 可以替代 useContext
  const theme = use(ThemeContext);

  return <button style={{ color: theme.color }}>按钮</button>;
}

function App() {
  return (
    <ThemeContext.Provider value={{ color: "blue" }}>
      <ThemedButton />
    </ThemeContext.Provider>
  );
}
```

### 5.3 use() 处理错误

```jsx
function UserProfile({ userPromise }) {
  // use() 可以在 try/catch 中使用，或由 ErrorBoundary 捕获
  const user = use(userPromise);

  return <div>{user.name}</div>;
}

// 配合 ErrorBoundary 使用
function App() {
  const userPromise = fetchUser(1);

  return (
    <ErrorBoundary fallback={<div>加载用户失败</div>}>
      <Suspense fallback={<div>加载中...</div>}>
        <UserProfile userPromise={userPromise} />
      </Suspense>
    </ErrorBoundary>
  );
}
```

### 5.4 条件使用 use()

```jsx
function UserCard({ shouldLoad, userPromise }) {
  const [showDetails, setShowDetails] = useState(false);

  // use() 可以在条件语句和循环中使用！
  // 这是 use() 相比 useContext 的一大优势
  const user = showDetails ? use(userPromise) : null;

  return (
    <div>
      <button onClick={() => setShowDetails(true)}>显示详情</button>
      {user && <div>{user.name}</div>}
    </div>
  );
}
```

---

## 6. Suspense 边界策略

### 6.1 边界放置原则

```
全局 Suspense 边界（最粗糙）：
  <Suspense fallback={<FullPageSpinner />}>
    <App />
  </Suspense>

路由级 Suspense 边界（中等粒度）：
  <Suspense fallback={<PageSkeleton />}>
    <Route path="/dashboard" element={<Dashboard />} />
  </Suspense>

组件级 Suspense 边界（最精细）：
  <div>
    <Suspense fallback={<HeaderSkeleton />}>
      <Header />
    </Suspense>
    <Suspense fallback={<ContentSkeleton />}>
      <Content />
    </Suspense>
    <Suspense fallback={<SidebarSkeleton />}>
      <Sidebar />
    </Suspense>
  </div>
```

### 6.2 嵌套 Suspense 边界

```jsx
function App() {
  return (
    // 外层边界：整页 fallback
    <Suspense fallback={<PageSkeleton />}>
      <Header />

      <div className="layout">
        {/* 内层边界：侧边栏独立加载 */}
        <Suspense fallback={<SidebarSkeleton />}>
          <Sidebar />
        </Suspense>

        {/* 内层边界：主内容区独立加载 */}
        <Suspense fallback={<ContentSkeleton />}>
          <MainContent />
        </Suspense>
      </div>
    </Suspense>
  );
}
```

### 6.3 最佳实践

```
推荐的边界策略：

1. 路由级边界：每个路由一个 Suspense 边界
2. 列表级边界：数据列表独立边界，不阻塞整体页面
3. 独立 widget 边界：评论区、推荐等独立模块单独边界
4. 避免过粗：整个应用只有一个边界，会导致任何数据加载都显示全屏 loading
5. 避免过细：每个小元素都加边界，会导致布局抖动

错误示范：
  - 整个 <App /> 只包一个 Suspense（太粗，任何加载都全屏 loading）
  - 每个文字标签都包 Suspense（太细，完全没必要）
```

---

## 7. Fallback UI 设计

### 7.1 骨架屏 (Skeleton)

```jsx
// 页面骨架屏
function PageSkeleton() {
  return (
    <div className="skeleton-page">
      <div className="skeleton-header" />
      <div className="skeleton-content">
        <div className="skeleton-line" />
        <div className="skeleton-line short" />
        <div className="skeleton-line" />
      </div>
    </div>
  );
}

// 卡片骨架屏
function CardSkeleton() {
  return (
    <div className="skeleton-card">
      <div className="skeleton-avatar" />
      <div className="skeleton-text">
        <div className="skeleton-line" />
        <div className="skeleton-line short" />
      </div>
    </div>
  );
}

// 表格骨架屏
function TableSkeleton({ rows = 5 }) {
  return (
    <table>
      <thead>
        <tr>
          {[1, 2, 3, 4].map((i) => (
            <th key={i}><div className="skeleton-line" /></th>
          ))}
        </tr>
      </thead>
      <tbody>
        {Array.from({ length: rows }).map((_, i) => (
          <tr key={i}>
            {[1, 2, 3, 4].map((j) => (
              <td key={j}><div className="skeleton-line" /></td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
```

### 7.2 带延迟的 Fallback

```jsx
import { Suspense, useState, useEffect } from "react";

// 延迟显示 loading，避免闪烁
function DelayedFallback({ delay = 200, children }) {
  const [show, setShow] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setShow(true), delay);
    return () => clearTimeout(timer);
  }, [delay]);

  if (!show) return null;
  return children;
}

function App() {
  return (
    <Suspense
      fallback={
        <DelayedFallback delay={300}>
          <Spinner />
        </DelayedFallback>
      }
    >
      <UserProfile />
    </Suspense>
  );
}
```

### 7.3 渐进式加载

```jsx
function App() {
  return (
    <Suspense fallback={<HeaderSkeleton />}>
      <Header />

      <Suspense fallback={<MainContentSkeleton />}>
        <MainContent>
          <Suspense fallback={<CommentsSkeleton />}>
            <Comments />
          </Suspense>
        </MainContent>
      </Suspense>
    </Suspense>
  );
}
// 渲染顺序：
// 1. HeaderSkeleton → Header（头部先加载完成）
// 2. MainContentSkeleton → MainContent（内容区加载完成）
// 3. CommentsSkeleton → Comments（评论区最后加载完成）
// 用户看到的是渐进式的内容出现，而不是整体白屏
```

---

## 8. 流式 SSR (Streaming SSR)

### 8.1 传统 SSR vs 流式 SSR

```
传统 SSR：
  服务器等待所有数据加载完成 → 生成完整 HTML → 发送给客户端 → 客户端渲染
  问题：TTFB (Time To First Byte) 很慢

流式 SSR：
  服务器立即开始发送 HTML → 流式传输已准备好的部分 → Suspense 部分发送 fallback
  → 数据到达后发送补充 HTML → 客户端逐步更新
  优势：TTFB 快，用户更早看到内容
```

### 8.2 React 18 流式 SSR

```jsx
// 服务端 (server.js)
import { renderToPipeableStream } from "react-dom/server";

app.get("/", (req, res) => {
  const { pipe } = renderToPipeableStream(
    <App />,
    {
      bootstrapScripts: ["/static/client.js"],
      onShellReady() {
        // Shell 就绪后立即开始流式传输
        res.setHeader("Content-Type", "text/html");
        pipe(res);
      },
      onShellError(error) {
        res.status(500).send("<h1>服务器错误</h1>");
      },
    }
  );
});
```

### 8.3 服务端组件中的 Suspense

```jsx
// App.jsx - 混合使用同步和异步组件
import { Suspense } from "react";

function App() {
  return (
    <html>
      <body>
        <Header />  {/* 同步组件，立即渲染 */}

        <Suspense fallback={<ContentSkeleton />}>
          <AsyncContent />  {/* 异步组件，数据加载中时显示骨架屏 */}
        </Suspense>

        <Suspense fallback={<CommentsSkeleton />}>
          <AsyncComments />
        </Suspense>
      </body>
    </html>
  );
}

// AsyncContent.jsx
async function AsyncContent() {
  const data = await fetchData(); // 服务端 await
  return <div>{data.title}</div>;
}
```

### 8.4 Next.js App Router 中的 Suspense

```jsx
// app/page.tsx
import { Suspense } from "react";

// 这个组件可以流式传输
export default function Page() {
  return (
    <div>
      <h1>Dashboard</h1>

      {/* 每个 Suspense 边界独立流式传输 */}
      <Suspense fallback={<StatsSkeleton />}>
        <Stats />
      </Suspense>

      <Suspense fallback={<ActivitySkeleton />}>
        <RecentActivity />
      </Suspense>
    </div>
  );
}

// app/components/Stats.tsx
async function Stats() {
  const stats = await fetch("/api/stats").then((r) => r.json());
  return <div>总用户: {stats.totalUsers}</div>;
}

// app/components/RecentActivity.tsx
async function RecentActivity() {
  const activity = await fetch("/api/activity").then((r) => r.json());
  return <ActivityList data={activity} />;
}
```

---

## 9. 完整实战示例

### 9.1 结合 Suspense 的数据获取架构

```jsx
import { Suspense, use, useState } from "react";

// 1. 数据获取函数
async function fetchUser(id) {
  const res = await fetch(`/api/users/${id}`);
  if (!res.ok) throw new Error("用户不存在");
  return res.json();
}

async function fetchPosts(userId) {
  const res = await fetch(`/api/users/${userId}/posts`);
  if (!res.ok) throw new Error("获取文章失败");
  return res.json();
}

// 2. 使用 use() 的组件
function UserHeader({ userPromise }) {
  const user = use(userPromise);
  return (
    <header>
      <img src={user.avatar} alt={user.name} />
      <h1>{user.name}</h1>
      <p>{user.bio}</p>
    </header>
  );
}

function PostList({ postsPromise }) {
  const posts = use(postsPromise);
  return (
    <ul>
      {posts.map((post) => (
        <li key={post.id}>
          <h2>{post.title}</h2>
          <p>{post.excerpt}</p>
        </li>
      ))}
    </ul>
  );
}

// 3. 主页面
function UserProfilePage({ userId }) {
  const [currentUserId] = useState(userId);

  // 创建 Promises
  const userPromise = fetchUser(currentUserId);
  const postsPromise = fetchPosts(currentUserId);

  return (
    <div>
      {/* 用户信息立即开始加载 */}
      <Suspense fallback={<UserHeaderSkeleton />}>
        <UserHeader userPromise={userPromise} />
      </Suspense>

      {/* 文章列表独立加载 */}
      <Suspense fallback={<PostListSkeleton />}>
        <PostList postsPromise={postsPromise} />
      </Suspense>
    </div>
  );
}
```

---

## 小结

- Suspense 是 React 的声明式异步处理机制，让组件可以在渲染时"暂停"等待数据
- `React.lazy()` 配合 Suspense 实现代码分割和懒加载
- Suspense 的核心机制是组件抛出 Promise，React 捕获后显示 fallback
- `use()` Hook (React 19) 简化了在组件中读取 Promise/Context 的方式
- 合理设计 Suspense 边界的粒度，平衡用户体验和复杂度
- 流式 SSR 利用 Suspense 实现渐进式 HTML 传输，改善 TTFB
- Fallback UI 应使用骨架屏而非空白 loading，提供更好的用户体验
