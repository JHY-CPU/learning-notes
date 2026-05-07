# useEffect 完全指南

## 目录

1. [基础概念](#基础概念)
2. [Effect 的执行时机](#effect-的执行时机)
3. [依赖数组规则](#依赖数组规则)
4. [Cleanup 的执行时机](#cleanup-的执行时机)
5. [Strict Mode 中的 Effect](#strict-mode-中的-effect)
6. [常见模式与实践](#常见模式与实践)
7. [useEffectEvent（RFC 概念）](#useeffecteventrfc-概念)
8. [useEffect vs useLayoutEffect](#useeffect-vs-uselayouteffect)

---

## 基础概念

`useEffect` 用于处理**副作用**（Side Effects）：数据获取、订阅、DOM 操作、日志记录等。

### 基本语法

```jsx
import { useEffect } from 'react';

function MyComponent() {
  // 1. 无依赖：每次渲染后执行
  useEffect(() => {
    console.log('组件已渲染');
  });

  // 2. 空依赖：只在挂载时执行一次
  useEffect(() => {
    console.log('组件已挂载');
  }, []);

  // 3. 有依赖：依赖变化时执行
  useEffect(() => {
    console.log(`count 变为: ${count}`);
  }, [count]);

  // 4. 带 cleanup 函数
  useEffect(() => {
    const subscription = subscribe();
    return () => {
      subscription.unsubscribe(); // 清理
    };
  }, []);

  return <div>...</div>;
}
```

---

## Effect 的执行时机

### 执行流程

```
组件函数体执行（渲染）
    │
    ▼
React 更新 DOM
    │
    ▼
浏览器绘制到屏幕（Paint）
    │
    ▼
useEffect 执行 ← 异步，在绘制之后
    │
    ▼
下次渲染前：先执行上一次的 cleanup
    │
    ▼
执行新的 effect
```

### 关键特点

```jsx
function App() {
  const [count, setCount] = useState(0);

  // effect 在浏览器绘制后异步执行，不会阻塞页面渲染
  useEffect(() => {
    // 这里的代码不会阻塞用户看到页面
    // 用户已经看到更新后的 DOM 了
    expensiveOperation();
  }, [count]);

  // 如果需要同步操作 DOM（在浏览器绘制前），使用 useLayoutEffect
  return <div>{count}</div>;
}
```

### 执行顺序示例

```jsx
function App() {
  const [val, setVal] = useState(0);

  console.log('1. 组件渲染');

  useEffect(() => {
    console.log('3. Effect 执行（在绘制之后）');
    return () => {
      console.log('4. Cleanup（下次 effect 前执行）');
    };
  }, [val]);

  return (
    <button onClick={() => setVal(v => v + 1)}>
      值: {val}
    </button>
  );
}

// 首次渲染:
// "1. 组件渲染"
// "3. Effect 执行（在绘制之后）"

// 点击按钮后:
// "1. 组件渲染"
// "4. Cleanup（下次 effect 前执行）"
// "3. Effect 执行（在绘制之后）"
```

---

## 依赖数组规则

### 规则一：包含所有在 effect 中使用的外部值

```jsx
function SearchResults({ query }) {
  const [results, setResults] = useState([]);

  // ✅ 正确：query 是 effect 中使用的外部值，必须列入依赖
  useEffect(() => {
    fetchResults(query).then(setResults);
  }, [query]);

  return <ul>{results.map(r => <li key={r.id}>{r.name}</li>)}</ul>;
}
```

### 规则二：闭包陷阱（Stale Closure）

```jsx
function Counter() {
  const [count, setCount] = useState(0);

  // ❌ 问题：effect 只在挂载时执行一次，闭包捕获了初始的 count = 0
  useEffect(() => {
    const timer = setInterval(() => {
      console.log(count); // 永远是 0！
      setCount(count + 1); // 永远设置为 1
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  // ✅ 方案一：使用函数式更新
  useEffect(() => {
    const timer = setInterval(() => {
      setCount(prev => prev + 1); // 不依赖外部 count
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  // ✅ 方案二：把 count 加入依赖
  useEffect(() => {
    const timer = setTimeout(() => {
      setCount(count + 1); // count 变化时会重新设置
    }, 1000);
    return () => clearTimeout(timer);
  }, [count]); // 注意：这会导致每次 count 变化都重新设置定时器

  return <p>计数: {count}</p>;
}
```

### 规则三：对象和数组依赖

```jsx
function UserProfile({ user }) {
  // ❌ 问题：user 是对象，每次父组件渲染时都是新引用
  // 即使内容没变，effect 也会执行
  useEffect(() => {
    fetchUserDetails(user.id);
  }, [user]); // user 对象引用每次不同

  // ✅ 方案一：只依赖需要的属性
  useEffect(() => {
    fetchUserDetails(user.id);
  }, [user.id]); // 基本类型，引用稳定

  // ✅ 方案二：使用 useMemo 稳定引用
  const stableUser = useMemo(() => user, [user.id, user.name]);

  useEffect(() => {
    fetchUserDetails(stableUser.id);
  }, [stableUser]);
}
```

### 规则四：函数依赖

```jsx
function App() {
  const [data, setData] = useState([]);

  // ❌ 问题：handleSuccess 每次渲染都是新函数
  const handleSuccess = (result) => {
    setData(result);
  };

  useEffect(() => {
    fetchData(handleSuccess);
  }, [handleSuccess]); // 每次都变，effect 每次都执行

  // ✅ 方案一：用 useCallback 稳定函数引用
  const handleSuccess = useCallback((result) => {
    setData(result);
  }, []);

  useEffect(() => {
    fetchData(handleSuccess);
  }, [handleSuccess]);

  // ✅ 方案二：把函数逻辑放进 effect 内部
  useEffect(() => {
    const handleSuccess = (result) => {
      setData(result);
    };
    fetchData(handleSuccess);
  }, []); // 不需要外部函数依赖

  return <div>...</div>;
}
```

### 依赖数组总结

| 情况 | 处理方式 |
|------|---------|
| 基本类型（string, number, boolean） | 直接列入依赖 |
| 对象/数组 | 依赖具体属性，或用 `useMemo` 稳定引用 |
| 函数 | 放入 effect 内部，或用 `useCallback` 稳定引用 |
| setState（setCount 等） | **不需要**列入依赖（React 保证引用稳定） |
| useRef | **不需要**列入依赖（`ref.current` 不参与依赖检查） |

---

## Cleanup 的执行时机

### 执行流程

```
首次渲染 → effect 执行 → ...
    │
组件更新（依赖变化）
    │
    ▼
cleanup（上一次的 effect 返回的函数）执行
    │
    ▼
新的 effect 执行
    │
组件卸载
    │
    ▼
cleanup 执行
```

### 示例：订阅与取消订阅

```jsx
function ChatRoom({ roomId }) {
  const [messages, setMessages] = useState([]);

  useEffect(() => {
    // effect：建立连接
    const connection = createConnection(roomId);
    connection.on('message', (msg) => {
      setMessages(prev => [...prev, msg]);
    });
    connection.connect();

    // cleanup：断开连接
    return () => {
      connection.disconnect();
    };
  }, [roomId]); // roomId 变化时：先断开旧连接，再建立新连接

  return (
    <ul>
      {messages.map((msg, i) => <li key={i}>{msg.text}</li>)}
    </ul>
  );
}
```

### 连续快速更新时的 cleanup

```
roomId = 'general' 渲染
    │ effect: 连接到 general
    │
roomId = 'random' 渲染
    │ cleanup: 断开 general
    │ effect: 连接到 random
    │
roomId = 'tech' 渲染
    │ cleanup: 断开 random
    │ effect: 连接到 tech
```

> cleanup 确保每次只保持一个活跃的连接，避免内存泄漏。

---

## Strict Mode 中的 Effect

React 18 的 Strict Mode 会在开发环境中**故意双重调用** effect，以帮助发现缺少 cleanup 的问题。

### 行为对比

```
普通模式（生产环境）:
    挂载 → effect
    更新 → cleanup → effect
    卸载 → cleanup

Strict Mode（开发环境）:
    挂载 → effect → cleanup → effect    （故意调用两次）
    更新 → cleanup → effect → cleanup → effect
    卸载 → cleanup
```

### 编写对 Strict Mode 友好的 Effect

```jsx
// ✅ 正确：有完整的 cleanup
useEffect(() => {
  const controller = new AbortController();

  fetch('/api/data', { signal: controller.signal })
    .then(res => res.json())
    .then(setData)
    .catch(err => {
      if (err.name !== 'AbortError') {
        setError(err);
      }
    });

  return () => controller.abort(); // cleanup 取消请求
}, []);

// ❌ 错误：没有 cleanup，Strict Mode 下会发起两次请求
useEffect(() => {
  fetch('/api/data')
    .then(res => res.json())
    .then(setData);
}, []);
```

---

## 常见模式与实践

### 模式一：数据获取（Data Fetching）

```jsx
function UserProfile({ userId }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false; // 竞态条件防护
    setLoading(true);

    fetch(`/api/users/${userId}`)
      .then(res => {
        if (!res.ok) throw new Error('请求失败');
        return res.json();
      })
      .then(data => {
        if (!cancelled) {
          setUser(data);
          setLoading(false);
        }
      })
      .catch(err => {
        if (!cancelled) {
          setError(err.message);
          setLoading(false);
        }
      });

    return () => {
      cancelled = true; // userId 变化时，忽略上一次请求的结果
    };
  }, [userId]);

  if (loading) return <p>加载中...</p>;
  if (error) return <p>错误: {error}</p>;
  return <p>{user?.name}</p>;
}
```

### 模式二：AbortController 取消请求

```jsx
function SearchResults({ query }) {
  const [results, setResults] = useState([]);

  useEffect(() => {
    if (!query) return;

    const controller = new AbortController();

    fetch(`/api/search?q=${query}`, { signal: controller.signal })
      .then(res => res.json())
      .then(data => setResults(data.results))
      .catch(err => {
        if (err.name !== 'AbortError') {
          console.error('搜索失败:', err);
        }
      });

    return () => controller.abort();
  }, [query]);

  return results.map(r => <div key={r.id}>{r.title}</div>);
}
```

### 模式三：事件订阅

```jsx
function WindowSize() {
  const [size, setSize] = useState({
    width: window.innerWidth,
    height: window.innerHeight,
  });

  useEffect(() => {
    const handleResize = () => {
      setSize({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return <p>窗口大小: {size.width} x {size.height}</p>;
}
```

### 模式四：DOM 测量

```jsx
function MeasuredDiv() {
  const ref = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  useEffect(() => {
    if (!ref.current) return;

    const observer = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect;
      setDimensions({ width, height });
    });

    observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);

  return (
    <div ref={ref}>
      尺寸: {dimensions.width.toFixed(0)} x {dimensions.height.toFixed(0)}
    </div>
  );
}
```

### 模式五：埋点/分析

```jsx
function PageView({ pageName }) {
  useEffect(() => {
    // 页面浏览上报
    analytics.track('page_view', {
      page: pageName,
      timestamp: Date.now(),
    });
    // 无 cleanup：纯汇报型 effect
  }, [pageName]);

  return <h1>{pageName}</h1>;
}
```

### 模式六：定时器

```jsx
function Timer() {
  const [seconds, setSeconds] = useState(0);
  const [isRunning, setIsRunning] = useState(false);

  useEffect(() => {
    if (!isRunning) return; // 不启动定时器

    const timer = setInterval(() => {
      setSeconds(prev => prev + 1);
    }, 1000);

    return () => clearInterval(timer); // 停止或卸载时清除
  }, [isRunning]);

  return (
    <div>
      <p>{seconds} 秒</p>
      <button onClick={() => setIsRunning(!isRunning)}>
        {isRunning ? '暂停' : '开始'}
      </button>
    </div>
  );
}
```

---

## useEffectEvent（RFC 概念）

> **注意**：`useEffectEvent` 目前是 RFC 阶段（React 实验性 API），尚未正式发布。这里介绍其设计理念，帮助理解 React 团队解决 effect 中"有时想要最新值、有时想要固定值"的思路。

### 问题场景

```jsx
function Chat({ roomId, theme }) {
  // effect 中需要最新的 roomId，但不需要在 roomId 变化时重新连接
  useEffect(() => {
    const connection = createConnection(roomId);
    connection.on('connected', () => {
      showNotification(`已连接到 ${roomId}，主题: ${theme}`);
      // 需要最新的 theme，但不想因为 theme 变化而重连
    });
    connection.connect();
    return () => connection.disconnect();
  }, [roomId]); // ❌ theme 不在依赖中会触发 lint 警告
}
```

### useEffectEvent 解决方案

```jsx
import { useEffectEvent } from 'react'; // 实验性 API

function Chat({ roomId, theme }) {
  // useEffectEvent 创建一个"始终获取最新值"的函数
  // 但它的引用是稳定的，不需要列入 effect 依赖
  const onConnected = useEffectEvent(() => {
    showNotification(`已连接到 ${roomId}，主题: ${theme}`);
    // roomId 和 theme 始终是最新值
  });

  useEffect(() => {
    const connection = createConnection(roomId);
    connection.on('connected', onConnected);
    connection.connect();
    return () => connection.disconnect();
  }, [roomId]); // onConnected 不需要列入依赖
  // roomId 变化时重连，theme 变化不重连但通知内容会用最新 theme
}
```

### 核心思想

- **Effect 依赖**：决定 effect **何时重新执行**
- **Effect Event**：在 effect 执行期间，获取**最新的 props/state 值**

```
useEffect(() => {
  // 依赖中的值 → effect 重新执行的触发条件
  // EffectEvent 中的值 → 始终最新，不触发重新执行
}, [依赖列表]);
```

---

## useEffect vs useLayoutEffect

### 时序对比

```
React 更新 DOM
    │
    ▼
useLayoutEffect ← 同步执行，在浏览器绘制之前
    │
    ▼
浏览器绘制（Paint）
    │
    ▼
useEffect ← 异步执行，在浏览器绘制之后
```

### useLayoutEffect 的使用场景

```jsx
import { useLayoutEffect, useRef, useState } from 'react';

// 场景一：测量 DOM 后立即更新（避免闪烁）
function Tooltip({ text, targetRef }) {
  const tooltipRef = useRef(null);
  const [position, setPosition] = useState({ top: 0, left: 0 });

  // ❌ 用 useEffect 会闪烁：先渲染在错误位置，再跳到正确位置
  // ✅ 用 useLayoutEffect：在绘制前算好位置，用户看不到闪烁
  useLayoutEffect(() => {
    if (!targetRef.current || !tooltipRef.current) return;

    const targetRect = targetRef.current.getBoundingClientRect();
    const tooltipRect = tooltipRef.current.getBoundingClientRect();

    setPosition({
      top: targetRect.top - tooltipRect.height - 8,
      left: targetRect.left + (targetRect.width - tooltipRect.width) / 2,
    });
  }, [text, targetRef]);

  return (
    <div
      ref={tooltipRef}
      style={{
        position: 'fixed',
        top: position.top,
        left: position.left,
      }}
    >
      {text}
    </div>
  );
}

// 场景二：同步读取 DOM 属性并更新
function ScrollRestoration() {
  const ref = useRef(null);

  useLayoutEffect(() => {
    // 在用户看到页面之前恢复滚动位置
    const savedScroll = sessionStorage.getItem('scroll');
    if (savedScroll && ref.current) {
      ref.current.scrollTop = parseInt(savedScroll, 10);
    }
  }, []);

  return <div ref={ref} style={{ height: '400px', overflow: 'auto' }}>...</div>;
}
```

### 选择指南

| 场景 | 使用 |
|------|------|
| 数据获取、订阅、日志 | `useEffect` |
| DOM 测量后触发新的渲染 | `useLayoutEffect` |
| 需要在绘制前同步更新样式 | `useLayoutEffect` |
| 防止用户看到中间状态/闪烁 | `useLayoutEffect` |
| 大多数副作用 | `useEffect`（默认选择） |

> **性能提示**：`useLayoutEffect` 是同步的，会阻塞浏览器绘制。大量使用会影响性能，只在确实需要时才使用。大多数场景用 `useEffect` 即可。

---

## 常见错误总结

| 错误 | 问题 | 正确做法 |
|------|------|---------|
| 缺少依赖 | 闭包拿到旧值 | 列出所有使用的外部值 |
| 对象/数组直接作依赖 | 每次渲染引用不同 | 依赖基本属性，或 `useMemo` |
| 没有 cleanup | 内存泄漏 | 返回 cleanup 函数 |
| 在 effect 中更新不需要的依赖 | 无限循环 | 用 `useCallback` / `useMemo` 稳定引用 |
| `async` 直接用于 effect 回调 | effect 返回值被当成 cleanup | 在 effect 内部定义 async 函数 |

```jsx
// ❌ 错误：effect 回调不能是 async
useEffect(async () => {
  const data = await fetchData();
  setData(data);
}, []);

// ✅ 正确：在 effect 内部定义 async 函数
useEffect(() => {
  let cancelled = false;

  async function loadData() {
    const data = await fetchData();
    if (!cancelled) setData(data);
  }

  loadData();
  return () => { cancelled = true; };
}, []);
```
