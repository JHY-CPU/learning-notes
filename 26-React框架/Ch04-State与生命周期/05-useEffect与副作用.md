# useEffect 与副作用（基础）

## 1. 什么是副作用（Side Effects）

在 React 中，**副作用**是指组件渲染过程之外的操作。纯渲染应该是没有副作用的——相同的输入产生相同的输出。

常见的副作用：
- API 请求（数据获取）
- 订阅（WebSocket、事件监听）
- 定时器（`setTimeout`、`setInterval`）
- DOM 操作（修改标题、焦点管理）
- 本地存储读写（`localStorage`）
- 日志记录

```jsx
// 纯渲染（无副作用）
function Greeting({ name }) {
  return <h1>Hello, {name}!</h1>;
}

// 有副作用
function UserFetcher({ userId }) {
  const [user, setUser] = React.useState(null);

  React.useEffect(() => {
    // 副作用：API 请求
    fetch(`/api/users/${userId}`)
      .then(res => res.json())
      .then(setUser);
  }, [userId]);

  return <div>{user?.name}</div>;
}
```

---

## 2. useEffect 基础

### 2.1 语法

```jsx
React.useEffect(() => {
  // 副作用代码
  // 在组件渲染后执行

  return () => {
    // 清理函数（可选）
    // 在组件卸载或下次 effect 执行前执行
  };
}, [dependencies]);  // 依赖数组
```

### 2.2 执行时机

```
组件渲染
    ↓
DOM 更新
    ↓
浏览器绘制
    ↓
useEffect 执行（异步）
```

`useEffect` 中的代码**不会阻塞浏览器绘制**，这是它与 `useLayoutEffect` 的核心区别。

### 2.3 基本示例

```jsx
function DocumentTitle({ title }) {
  React.useEffect(() => {
    // 每次 title 变化后，更新文档标题
    document.title = title;
  }, [title]);

  return <h1>{title}</h1>;
}
```

---

## 3. 依赖数组（Dependency Array）

依赖数组决定了 effect 何时重新执行。

### 3.1 三种情况

```jsx
// 情况一：无依赖数组——每次渲染后都执行
React.useEffect(() => {
  console.log('Every render');
});

// 情况二：空依赖数组 []——只在挂载时执行一次
React.useEffect(() => {
  console.log('Only on mount');
}, []);

// 情况三：有依赖项——依赖变化时执行
React.useEffect(() => {
  console.log('userId changed:', userId);
}, [userId]);
```

### 3.2 依赖的规则

Effect 中**使用到的所有响应式值**都应该放入依赖数组：

```jsx
function SearchResults({ query, page }) {
  const [results, setResults] = React.useState([]);

  React.useEffect(() => {
    fetch(`/api/search?q=${query}&page=${page}`)
      .then(res => res.json())
      .then(setResults);
  }, [query, page]);  // query 和 page 都是依赖

  return results.map(r => <div key={r.id}>{r.name}</div>);
}
```

### 3.3 响应式值包括

- Props
- State
- 在组件体内定义的函数或变量（每次渲染都是新引用）

```jsx
function Component({ userId }) {
  const [token, setToken] = React.useState('');

  // options 每次渲染都是新对象
  const options = { headers: { Authorization: token } };

  // 这个函数每次渲染都是新的
  const processData = (data) => data.filter(Boolean);

  React.useEffect(() => {
    // 使用了 userId, token, options, processData
    // 所有这些都应该是依赖
    // 但 options 和 processData 每次都是新的，会导致无限循环！
    fetchData(userId, options).then(processData);
  }, [userId, token, options, processData]);  // 问题：无限循环
}
```

---

## 4. 清理函数（Cleanup Function）

清理函数用于取消订阅、清除定时器、移除事件监听器等，防止内存泄漏。

### 4.1 什么时候执行

- 组件**卸载**时
- 下一次 effect 执行**之前**（依赖变化时）

### 4.2 常见清理场景

```jsx
// 1. 清除定时器
function Timer() {
  const [count, setCount] = React.useState(0);

  React.useEffect(() => {
    const timer = setInterval(() => {
      setCount(c => c + 1);
    }, 1000);

    return () => clearInterval(timer);  // 清理
  }, []);

  return <div>{count}</div>;
}

// 2. 移除事件监听
function WindowSize() {
  const [width, setWidth] = React.useState(window.innerWidth);

  React.useEffect(() => {
    const handleResize = () => setWidth(window.innerWidth);
    window.addEventListener('resize', handleResize);

    return () => window.removeEventListener('resize', handleResize);  // 清理
  }, []);

  return <div>Width: {width}</div>;
}

// 3. 取消 WebSocket 连接
function ChatRoom({ roomId }) {
  React.useEffect(() => {
    const socket = new WebSocket(`wss://example.com/room/${roomId}`);

    socket.onmessage = (event) => {
      console.log('Message:', event.data);
    };

    return () => socket.close();  // 清理
  }, [roomId]);

  return <div>Chat: {roomId}</div>;
}

// 4. 取消 API 请求（AbortController）
function DataFetcher({ url }) {
  const [data, setData] = React.useState(null);

  React.useEffect(() => {
    const controller = new AbortController();

    fetch(url, { signal: controller.signal })
      .then(res => res.json())
      .then(setData)
      .catch(err => {
        if (err.name !== 'AbortError') {
          console.error(err);
        }
      });

    return () => controller.abort();  // 清理
  }, [url]);

  return <div>{data ? JSON.stringify(data) : 'Loading...'}</div>;
}
```

### 4.3 清理的重要性

不清理可能导致：
- 内存泄漏（定时器、订阅持续占用资源）
- 状态更新到已卸载的组件（React 警告）
- 事件监听器堆积（性能下降）

---

## 5. 常见使用模式

### 5.1 数据获取

```jsx
function UserProfile({ userId }) {
  const [user, setUser] = React.useState(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState(null);

  React.useEffect(() => {
    let cancelled = false;
    setLoading(true);

    fetch(`/api/users/${userId}`)
      .then(res => {
        if (!res.ok) throw new Error('Failed to fetch');
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

    return () => { cancelled = true; };
  }, [userId]);

  if (loading) return <Spinner />;
  if (error) return <Error message={error} />;
  return <div>{user.name}</div>;
}
```

### 5.2 外部系统订阅

```jsx
function OnlineStatus() {
  const [isOnline, setIsOnline] = React.useState(navigator.onLine);

  React.useEffect(() => {
    const goOnline = () => setIsOnline(true);
    const goOffline = () => setIsOnline(false);

    window.addEventListener('online', goOnline);
    window.addEventListener('offline', goOffline);

    return () => {
      window.removeEventListener('online', goOnline);
      window.removeEventListener('offline', goOffline);
    };
  }, []);

  return <div>{isOnline ? 'Online' : 'Offline'}</div>;
}
```

### 5.3 DOM 副作用

```jsx
function Modal({ isOpen, children }) {
  React.useEffect(() => {
    if (isOpen) {
      // 打开时禁止背景滚动
      document.body.style.overflow = 'hidden';
    }

    return () => {
      // 关闭时恢复
      document.body.style.overflow = 'unset';
    };
  }, [isOpen]);

  if (!isOpen) return null;

  return <div className="modal">{children}</div>;
}
```

### 5.4 与第三方库集成

```jsx
function Chart({ data }) {
  const chartRef = React.useRef(null);
  const instanceRef = React.useRef(null);

  React.useEffect(() => {
    // 创建图表实例
    instanceRef.current = new ChartLibrary(chartRef.current, {
      data,
    });

    return () => {
      // 销毁图表实例
      instanceRef.current.destroy();
    };
  }, []);  // 只创建一次

  React.useEffect(() => {
    // 数据变化时更新图表
    if (instanceRef.current) {
      instanceRef.current.setData(data);
    }
  }, [data]);

  return <div ref={chartRef} />;
}
```

---

## 6. 跳过 Effect

### 6.1 依赖不变时跳过

```jsx
function Example({ userId, token }) {
  React.useEffect(() => {
    // 只有当 userId 或 token 变化时才执行
    fetchData(userId, token);
  }, [userId, token]);

  return <div>...</div>;
}
```

### 6.2 使用 ref 避免不必要的依赖

```jsx
function Chat({ roomId }) {
  const [message, setMessage] = React.useState('');
  const latestMessage = React.useRef(message);

  // ref 始终是最新的值，但 ref 本身引用不变
  latestMessage.current = message;

  React.useEffect(() => {
    const timer = setInterval(() => {
      // 使用 ref 读取最新值，不需要把 message 放入依赖
      console.log('Latest message:', latestMessage.current);
    }, 5000);

    return () => clearInterval(timer);
  }, [roomId]);  // 只在 roomId 变化时重新创建定时器

  return <input value={message} onChange={e => setMessage(e.target.value)} />;
}
```

---

## 7. 无限循环陷阱

### 7.1 常见原因

```jsx
// 错误：对象/数组在渲染时创建，每次引用不同
function BadComponent({ userId }) {
  const options = { page: 1, limit: 10 };  // 每次渲染都是新对象

  React.useEffect(() => {
    fetchData(userId, options);
  }, [userId, options]);  // options 每次都变 → 无限循环！
}

// 修正：将 options 移到 effect 内部
function GoodComponent({ userId }) {
  React.useEffect(() => {
    const options = { page: 1, limit: 10 };  // 在 effect 内部创建
    fetchData(userId, options);
  }, [userId]);  // 只依赖 userId
}
```

### 7.2 setState 触发重新渲染

```jsx
// 错误：effect 中更新 state，state 变化又触发 effect
function BadComponent() {
  const [data, setData] = React.useState(null);

  React.useEffect(() => {
    setData(fetchData());  // setData 触发重新渲染 → effect 再次执行 → 无限循环
  });  // 没有依赖数组，每次渲染都执行
}

// 修正：添加依赖数组或条件判断
function GoodComponent() {
  const [data, setData] = React.useState(null);

  React.useEffect(() => {
    fetchData().then(result => setData(result));
  }, []);  // 空依赖，只执行一次
}
```

### 7.3 函数依赖

```jsx
// 错误：函数每次渲染都是新的引用
function BadComponent() {
  const processData = (data) => data.filter(Boolean);

  React.useEffect(() => {
    someOperation(processData);
  }, [processData]);  // processData 每次都变 → 无限循环
}

// 修正一：用 useCallback 缓存函数
function GoodComponent() {
  const processData = React.useCallback((data) => {
    return data.filter(Boolean);
  }, []);  // 空依赖，引用不变

  React.useEffect(() => {
    someOperation(processData);
  }, [processData]);
}

// 修正二：将函数移到 effect 内部
function GoodComponent() {
  React.useEffect(() => {
    const processData = (data) => data.filter(Boolean);
    someOperation(processData);
  }, []);
}
```

---

## 8. useEffect 执行顺序

### 8.1 多个 Effect 的执行顺序

```jsx
function App() {
  React.useEffect(() => {
    console.log('Effect 1');
    return () => console.log('Cleanup 1');
  });

  React.useEffect(() => {
    console.log('Effect 2');
    return () => console.log('Cleanup 2');
  });

  return <div>App</div>;
}

// 首次渲染: Effect 1 → Effect 2
// 更新时:   Cleanup 1 → Cleanup 2 → Effect 1 → Effect 2
// 卸载时:   Cleanup 1 → Cleanup 2
```

### 8.2 嵌套组件的执行顺序

```jsx
function Parent() {
  React.useEffect(() => {
    console.log('Parent effect');
    return () => console.log('Parent cleanup');
  });

  return <Child />;
}

function Child() {
  React.useEffect(() => {
    console.log('Child effect');
    return () => console.log('Child cleanup');
  });

  return <div>Child</div>;
}

// 挂载: Child effect → Parent effect（子组件先）
// 卸载: Parent cleanup → Child cleanup（父组件先）
```

---

## 9. 常见面试问题

### Q1：useEffect 的依赖数组是浅比较吗？

是的。React 使用 `Object.is()` 进行比较，对原始类型是值比较，对引用类型是引用比较。

### Q2：什么时候不需要依赖数组？

几乎不需要。如果省略依赖数组，effect 每次渲染后都执行。这通常不是你想要的。如果确实需要，确保 effect 中没有导致无限循环的操作。

### Q3：如何处理 async/await？

`useEffect` 的回调不能是 async 函数（因为它需要返回清理函数或 undefined）。应该在内部定义 async 函数：

```jsx
React.useEffect(() => {
  async function fetchData() {
    const response = await fetch(url);
    const data = await response.json();
    setData(data);
  }
  fetchData();
}, [url]);
```

### Q4：effect 和渲染的关系是什么？

Effect 在渲染之后执行，且可以访问渲染时的 props 和 state。Effect 中的 setState 会触发新的渲染。关键区别：渲染应该是纯的（无副作用），副作用放到 effect 中。
