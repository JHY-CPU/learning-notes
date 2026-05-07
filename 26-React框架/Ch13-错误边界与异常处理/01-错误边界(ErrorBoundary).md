# Ch13-01 错误边界 (ErrorBoundary)

## 目录

1. [错误边界概念](#1-错误边界概念)
2. [getDerivedStateFromError](#2-getderivedstatefromerror)
3. [componentDidCatch](#3-componentdidcatch)
4. [类组件实现](#4-类组件实现)
5. [错误边界放置策略](#5-错误边界放置策略)
6. [粒度策略](#6-粒度策略)
7. [Fallback UI 设计](#7-fallback-ui-设计)
8. [React 19 新特性](#8-react-19-新特性)

---

## 1. 错误边界概念

### 1.1 什么是错误边界

错误边界 (Error Boundary) 是 React 中一种特殊的组件，用于**捕获子组件树中渲染期间的 JavaScript 错误**，记录错误并显示备用 UI，而不是让整个组件树崩溃。

```
没有错误边界：
  App
   ├── Header       ✅ 正常
   ├── Sidebar      ✅ 正常
   └── Content      ❌ 出错 → 整个应用崩溃白屏

有错误边界：
  App
   ├── Header              ✅ 正常
   ├── ErrorBoundary
   │    └── Sidebar        ✅ 正常
   └── ErrorBoundary
        └── Content        ❌ 出错 → 显示 fallback，其余部分正常
```

### 1.2 错误边界能捕获的错误

```jsx
// ✅ 渲染中的错误
function BrokenComponent() {
  throw new Error("渲染出错");
}

// ✅ 生命周期方法中的错误
class BrokenClass extends React.Component {
  componentDidMount() {
    throw new Error("生命周期出错");
  }
}

// ✅ 构造函数中的错误
class BrokenConstructor extends React.Component {
  constructor() {
    super();
    throw new Error("构造函数出错");
  }
}
```

### 1.3 错误边界不能捕获的错误

```jsx
// ❌ 事件处理器中的错误
function Button() {
  const handleClick = () => {
    throw new Error("事件处理出错"); // 不会被 ErrorBoundary 捕获
  };
  return <button onClick={handleClick}>Click</button>;
}

// ❌ 异步代码中的错误
function AsyncComponent() {
  useEffect(() => {
    setTimeout(() => {
      throw new Error("异步出错"); // 不会被捕获
    }, 1000);
  }, []);
  return <div>Async</div>;
}

// ❌ 服务端渲染中的错误
// ❌ 错误边界组件自身的错误

// 对于以上错误，需要使用 try/catch 或 window.onerror
```

---

## 2. getDerivedStateFromError

### 2.1 作用

`static getDerivedStateFromError(error)` 在子组件抛出错误后被调用，用于更新 state 以显示 fallback UI：

```jsx
class ErrorBoundary extends React.Component {
  state = { hasError: false };

  static getDerivedStateFromError(error) {
    // 返回新的 state 来更新 UI
    return { hasError: true };
  }

  render() {
    if (this.state.hasError) {
      return <h1>出错了</h1>;
    }
    return this.props.children;
  }
}
```

### 2.2 特点

```
getDerivedStateFromError 的特点：
  1. 是静态方法，不能访问 this
  2. 在"渲染阶段"被调用
  3. 应该返回新的 state 对象
  4. 不应该有副作用（如发送日志）
  5. 子组件出错后会先卸载，然后 ErrorBoundary 重新渲染显示 fallback
```

---

## 3. componentDidCatch

### 3.1 作用

`componentDidCatch(error, errorInfo)` 在子组件抛出错误后被调用，用于执行副作用（如发送错误日志）：

```jsx
class ErrorBoundary extends React.Component {
  state = { hasError: false };

  componentDidCatch(error, errorInfo) {
    // 发送错误日志到监控服务
    logErrorToService({
      error: error.toString(),
      componentStack: errorInfo.componentStack,
      timestamp: Date.now(),
    });
  }

  render() {
    if (this.state.hasError) {
      return <h1>出错了</h1>;
    }
    return this.props.children;
  }
}
```

### 3.2 errorInfo 对象

```jsx
componentDidCatch(error, errorInfo) {
  // error: Error 对象
  console.log(error.message);
  console.log(error.stack);

  // errorInfo.componentStack: 组件调用栈
  // 包含从出错组件到 ErrorBoundary 的完整组件层级
  console.log(errorInfo.componentStack);
  // 示例输出：
  //     at BrokenComponent
  //     at div
  //     at Content
  //     at ErrorBoundary
  //     at App
}
```

---

## 4. 类组件实现

### 4.1 基础实现

```jsx
import React from "react";

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error) {
    // 更新 state 以显示 fallback UI
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    // 记录错误信息
    this.setState({ errorInfo });

    // 发送到监控服务
    console.error("ErrorBoundary 捕获到错误:", error, errorInfo);

    // 可以发送到 Sentry、LogRocket 等
    // Sentry.captureException(error, { extra: errorInfo });
  }

  render() {
    if (this.state.hasError) {
      // 使用自定义 fallback 或默认 fallback
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="error-boundary">
          <h2>出错了</h2>
          <details style={{ whiteSpace: "pre-wrap" }}>
            <summary>错误详情</summary>
            {this.state.error?.toString()}
            <br />
            {this.state.errorInfo?.componentStack}
          </details>
          <button onClick={() => this.setState({ hasError: false, error: null, errorInfo: null })}>
            重试
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
```

### 4.2 带重置功能的实现

```jsx
class ErrorBoundary extends React.Component {
  state = { hasError: false, error: null };

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    this.props.onError?.(error, errorInfo);
  }

  resetError = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      // 如果提供了 renderFallback 函数，使用它
      if (this.props.renderFallback) {
        return this.props.renderFallback({
          error: this.state.error,
          resetError: this.resetError,
        });
      }

      return this.props.fallback ?? <div>Something went wrong.</div>;
    }

    return this.props.children;
  }
}

// 使用
function App() {
  return (
    <ErrorBoundary
      renderFallback={({ error, resetError }) => (
        <div>
          <p>错误: {error.message}</p>
          <button onClick={resetError}>重试</button>
        </div>
      )}
      onError={(error, info) => {
        console.error(error);
      }}
    >
      <MyComponent />
    </ErrorBoundary>
  );
}
```

### 4.3 React 18 的 resetKeys 功能

```jsx
// 使用 resetKeys 在特定 prop 变化时自动重置错误状态
function ErrorBoundaryWithResetKeys({ children, resetKeys, ...props }) {
  const [hasError, setHasError] = useState(false);
  const prevResetKeys = useRef(resetKeys);

  // 如果 resetKeys 变化，重置错误状态
  useEffect(() => {
    if (
      hasError &&
      resetKeys?.some((key, i) => key !== prevResetKeys.current?.[i])
    ) {
      setHasError(false);
    }
    prevResetKeys.current = resetKeys;
  }, [hasError, resetKeys]);

  // 注意：实际的 ErrorBoundary 仍需类组件实现
  // 这是一个模式示意
  return (
    <ClassErrorBoundary
      {...props}
      hasError={hasError}
      onError={() => setHasError(true)}
    >
      {children}
    </ClassErrorBoundary>
  );
}

// 使用
<ErrorBoundaryWithResetKeys resetKeys={[userId]}>
  <UserProfile userId={userId} />
</ErrorBoundaryWithResetKeys>
```

---

## 5. 错误边界放置策略

### 5.1 路由级错误边界

```jsx
// 每个路由都有独立的错误边界
function App() {
  return (
    <Router>
      <ErrorBoundary fallback={<GlobalError />}>
        <Routes>
          <Route
            path="/"
            element={
              <ErrorBoundary fallback={<HomeError />}>
                <Home />
              </ErrorBoundary>
            }
          />
          <Route
            path="/dashboard"
            element={
              <ErrorBoundary fallback={<DashboardError />}>
                <Dashboard />
              </ErrorBoundary>
            }
          />
        </Routes>
      </ErrorBoundary>
    </Router>
  );
}
```

### 5.2 组件级错误边界

```jsx
function Dashboard() {
  return (
    <div className="dashboard">
      <ErrorBoundary fallback={<WidgetError name="用户统计" />}>
        <UserStats />
      </ErrorBoundary>

      <ErrorBoundary fallback={<WidgetError name="销售数据" />}>
        <SalesChart />
      </ErrorBoundary>

      <ErrorBoundary fallback={<WidgetError name="通知" />}>
        <Notifications />
      </ErrorBoundary>
    </div>
  );
}
// 销售数据组件出错时，用户统计和通知仍正常显示
```

### 5.3 第三方组件隔离

```jsx
// 将不稳定的第三方组件包在 ErrorBoundary 中
function App() {
  return (
    <div>
      <ErrorBoundary fallback={<div>图表加载失败</div>}>
        <ThirdPartyChart data={chartData} />
      </ErrorBoundary>

      <ErrorBoundary fallback={<div>地图加载失败</div>}>
        <ThirdPartyMap />
      </ErrorBoundary>
    </div>
  );
}
```

---

## 6. 粒度策略

### 6.1 错误边界的层级

```
推荐的 ErrorBoundary 层级：

1. 全局 ErrorBoundary（兜底）
   └── 捕获所有未被更细粒度边界捕获的错误
   └── 显示通用错误页面

2. 路由级 ErrorBoundary
   └── 每个路由一个边界
   └── 路由出错时不影响导航和其他路由

3. 功能模块级 ErrorBoundary
   └── 独立功能模块各有一个边界
   └── 比如评论区、推荐模块、侧边栏

4. 第三方组件级 ErrorBoundary
   └── 隔离不稳定的第三方组件
```

### 6.2 避免过度使用

```jsx
// ❌ 过度使用 - 每个小元素都包 ErrorBoundary
function BadPractice() {
  return (
    <ErrorBoundary>
      <Title />
    </ErrorBoundary>
    <ErrorBoundary>
      <Subtitle />
    </ErrorBoundary>
    <ErrorBoundary>
      <Button />
    </ErrorBoundary>
  );
  // 问题：ErrorBoundary 本身有开销
  // 而且这些组件太小，出错概率低

// ✅ 合理使用
function GoodPractice() {
  return (
    <div>
      <Header />  {/* 头部很稳定，不需要 ErrorBoundary */}

      <ErrorBoundary fallback={<SidebarError />}>
        <Sidebar />  {/* 侧边栏可能加载外部数据 */}
      </ErrorBoundary>

      <ErrorBoundary fallback={<ContentError />}>
        <MainContent />  {/* 主内容区最重要，需要保护 */}
      </ErrorBoundary>
    </div>
  );
}
```

---

## 7. Fallback UI 设计

### 7.1 分级 Fallback

```jsx
// 全局级 Fallback - 最严重
function GlobalErrorFallback({ error, resetError }) {
  return (
    <div className="global-error">
      <h1>应用出现了问题</h1>
      <p>我们已经记录了这个错误，请稍后重试。</p>
      <button onClick={resetError}>重新加载</button>
      <a href="/">返回首页</a>
    </div>
  );
}

// 路由级 Fallback
function PageErrorFallback({ error, resetError }) {
  return (
    <div className="page-error">
      <h2>页面加载失败</h2>
      <p>{error.message}</p>
      <button onClick={resetError}>重试</button>
    </div>
  );
}

// 组件级 Fallback - 最轻量
function WidgetErrorFallback({ name, resetError }) {
  return (
    <div className="widget-error">
      <p>{name || "模块"}加载失败</p>
      <button onClick={resetError}>重试</button>
    </div>
  );
}
```

### 7.2 带重试的 Fallback

```jsx
function RetryableErrorFallback({ error, resetError, maxRetries = 3 }) {
  const [retryCount, setRetryCount] = useState(0);

  const handleRetry = () => {
    setRetryCount((c) => c + 1);
    resetError();
  };

  if (retryCount >= maxRetries) {
    return (
      <div>
        <p>多次重试失败，请联系客服</p>
        <a href="mailto:support@example.com">联系支持</a>
      </div>
    );
  }

  return (
    <div>
      <p>加载失败 ({retryCount}/{maxRetries})</p>
      <button onClick={handleRetry}>
        重试 ({maxRetries - retryCount} 次剩余)
      </button>
    </div>
  );
}
```

---

## 8. React 19 新特性

### 8.1 函数组件 ErrorBoundary (实验性)

React 19 正在探索函数组件的错误边界支持：

```jsx
// React 19 实验性 API（可能在未来版本稳定）
import { experimental_useErrorBoundary as useErrorBoundary } from "react";

// 目前推荐使用 react-error-boundary 库
import { ErrorBoundary } from "react-error-boundary";

function App() {
  return (
    <ErrorBoundary
      FallbackComponent={ErrorFallback}
      onError={(error, info) => console.error(error, info)}
      onReset={() => window.location.reload()}
    >
      <MyApp />
    </ErrorBoundary>
  );
}

function ErrorFallback({ error, resetErrorBoundary }) {
  return (
    <div>
      <h2>出错了</h2>
      <pre>{error.message}</pre>
      <button onClick={resetErrorBoundary}>重试</button>
    </div>
  );
}
```

### 8.2 react-error-boundary 库

```bash
npm install react-error-boundary
```

```jsx
import { ErrorBoundary } from "react-error-boundary";

function ErrorFallback({ error, resetErrorBoundary }) {
  return (
    <div role="alert">
      <p>出了点问题:</p>
      <pre>{error.message}</pre>
      <button onClick={resetErrorBoundary}>重试</button>
    </div>
  );
}

function App() {
  return (
    <ErrorBoundary
      FallbackComponent={ErrorFallback}
      onReset={(details) => {
        // 重置逻辑
      }}
      onError={(error, info) => {
        // 日志上报
      }}
    >
      <MyComponent />
    </ErrorBoundary>
  );
}

// 与 Suspense 配合
function App() {
  return (
    <ErrorBoundary FallbackComponent={ErrorFallback}>
      <Suspense fallback={<Loading />}>
        <AsyncComponent />
      </Suspense>
    </ErrorBoundary>
  );
}
```

---

## 小结

- 错误边界是 React 中捕获子组件渲染错误的机制
- 必须使用类组件实现（通过 `getDerivedStateFromError` 和 `componentDidCatch`）
- 错误边界不能捕获事件处理器、异步代码和自身的错误
- 合理的放置策略：全局级、路由级、组件级
- Fallback UI 应提供重试机制和用户友好的错误提示
- 推荐使用 `react-error-boundary` 库简化使用
