# useRef、useMemo 与 useCallback

## 目录

1. [useRef 基础](#useref-基础)
2. [useRef 获取 DOM 引用](#useref-获取-dom-引用)
3. [useRef 存储可变值](#useref-存储可变值)
4. [useMemo 缓存计算结果](#usememo-缓存计算结果)
5. [useMemo 保持引用稳定](#usememo-保持引用稳定)
6. [useCallback 缓存函数](#usecallback-缓存函数)
7. [何时不应该使用记忆化](#何时不应该使用记忆化)
8. [React.memo 包裹组件](#reactmemo-包裹组件)
9. [依赖数组最佳实践](#依赖数组最佳实践)

---

## useRef 基础

`useRef` 返回一个可变的 ref 对象，其 `.current` 属性被初始化为传入的参数。ref 对象在组件的整个生命周期中保持不变。

```jsx
import { useRef } from 'react';

function MyComponent() {
  const ref = useRef(initialValue);
  // ref.current 初始值为 initialValue
  // 修改 ref.current 不会触发重新渲染

  return <div ref={ref}>...</div>;
}
```

### useRef vs useState

| 特性 | useRef | useState |
|------|--------|----------|
| 修改值触发重新渲染 | 否 | 是 |
| 值在渲染间保持 | 是 | 是 |
| 可以绑定 DOM | 是 | 否 |
| 同步修改立即可见 | 是（修改 ref.current） | 否（setState 异步） |

---

## useRef 获取 DOM 引用

### 基本用法

```jsx
function TextInput() {
  const inputRef = useRef(null);

  const handleFocus = () => {
    inputRef.current.focus(); // 直接操作 DOM
  };

  return (
    <div>
      <input ref={inputRef} type="text" placeholder="输入内容" />
      <button onClick={handleFocus}>聚焦输入框</button>
    </div>
  );
}
```

### 常见 DOM 操作场景

```jsx
function ScrollableList() {
  const listRef = useRef(null);
  const itemRefs = useRef({});

  // 滚动到顶部
  const scrollToTop = () => {
    listRef.current.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // 滚动到指定项
  const scrollToItem = (id) => {
    itemRefs.current[id]?.scrollIntoView({ behavior: 'smooth' });
  };

  // 获取列表容器的尺寸
  const getListSize = () => {
    if (!listRef.current) return null;
    return {
      width: listRef.current.clientWidth,
      height: listRef.current.clientHeight,
      scrollTop: listRef.current.scrollTop,
    };
  };

  return (
    <div>
      <button onClick={scrollToTop}>回到顶部</button>
      <div ref={listRef} style={{ height: '400px', overflow: 'auto' }}>
        {items.map(item => (
          <div
            key={item.id}
            ref={el => { itemRefs.current[item.id] = el; }}
          >
            {item.name}
          </div>
        ))}
      </div>
    </div>
  );
}
```

### 视频/音频控制

```jsx
function VideoPlayer({ src }) {
  const videoRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);

  const togglePlay = () => {
    if (videoRef.current.paused) {
      videoRef.current.play();
      setIsPlaying(true);
    } else {
      videoRef.current.pause();
      setIsPlaying(false);
    }
  };

  const seek = (seconds) => {
    videoRef.current.currentTime += seconds;
  };

  return (
    <div>
      <video ref={videoRef} src={src} />
      <button onClick={togglePlay}>{isPlaying ? '暂停' : '播放'}</button>
      <button onClick={() => seek(-10)}>后退 10 秒</button>
      <button onClick={() => seek(10)}>前进 10 秒</button>
    </div>
  );
}
```

### Forwarding Refs（转发 Ref）

```jsx
import { forwardRef, useRef } from 'react';

// 使用 forwardRef 让父组件获取子组件内部 DOM 的引用
const CustomInput = forwardRef(function CustomInput(props, ref) {
  return (
    <div className="custom-input">
      <label>{props.label}</label>
      <input ref={ref} {...props} />
    </div>
  );
});

function Form() {
  const inputRef = useRef(null);

  return (
    <div>
      <CustomInput ref={inputRef} label="用户名" />
      <button onClick={() => inputRef.current.focus()}>聚焦</button>
    </div>
  );
}
```

---

## useRef 存储可变值

`useRef` 的一个强大用途是存储**不参与渲染**的可变值。

### 存储定时器 ID

```jsx
function Stopwatch() {
  const [time, setTime] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const timerRef = useRef(null);

  const start = () => {
    if (isRunning) return;
    setIsRunning(true);
    // 存储 timer ID，用于后续清除
    timerRef.current = setInterval(() => {
      setTime(prev => prev + 10);
    }, 10);
  };

  const stop = () => {
    setIsRunning(false);
    clearInterval(timerRef.current); // 使用 ref 中存储的 ID
  };

  const reset = () => {
    stop();
    setTime(0);
  };

  useEffect(() => {
    return () => clearInterval(timerRef.current); // 清理
  }, []);

  return (
    <div>
      <p>{(time / 1000).toFixed(2)} 秒</p>
      <button onClick={start} disabled={isRunning}>开始</button>
      <button onClick={stop} disabled={!isRunning}>停止</button>
      <button onClick={reset}>重置</button>
    </div>
  );
}
```

### 获取前一次的值

```jsx
function usePrevious(value) {
  const ref = useRef();

  useEffect(() => {
    ref.current = value; // 渲染后更新 ref
  });

  return ref.current; // 返回上一次的值
}

function Counter() {
  const [count, setCount] = useState(0);
  const prevCount = usePrevious(count);

  return (
    <div>
      <p>当前: {count}, 上一次: {prevCount ?? 'N/A'}</p>
      <p>变化: {prevCount != null ? count - prevCount : 0}</p>
      <button onClick={() => setCount(c => c + 1)}>+1</button>
    </div>
  );
}
```

### 存储最新值（避免 stale closure）

```jsx
function Chat({ onReceiveMessage }) {
  // 存储最新的回调函数
  const onReceiveMessageRef = useRef(onReceiveMessage);

  // 每次渲染后更新 ref 中的值
  useEffect(() => {
    onReceiveMessageRef.current = onReceiveMessage;
  });

  useEffect(() => {
    const socket = new WebSocket('wss://chat.example.com');

    socket.onmessage = (event) => {
      // 通过 ref 获取最新的回调，避免闭包捕获旧值
      onReceiveMessageRef.current(JSON.parse(event.data));
    };

    return () => socket.close();
  }, []); // 空依赖，不随 onReceiveMessage 变化而重连

  return <div>聊天室</div>;
}
```

### 防止重复执行

```jsx
function DataFetcher({ url }) {
  const [data, setData] = useState(null);
  const fetchedRef = useRef(new Set());

  useEffect(() => {
    // 防止重复请求同一 URL
    if (fetchedRef.current.has(url)) return;

    fetchedRef.current.add(url);
    fetch(url)
      .then(res => res.json())
      .then(setData);
  }, [url]);

  return data ? <pre>{JSON.stringify(data, null, 2)}</pre> : <p>加载中...</p>;
}
```

### 追踪组件是否已卸载

```jsx
function useIsMounted() {
  const isMountedRef = useRef(true);

  useEffect(() => {
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  return isMountedRef;
}

function AsyncComponent() {
  const [data, setData] = useState(null);
  const isMounted = useIsMounted();

  useEffect(() => {
    fetch('/api/data')
      .then(res => res.json())
      .then(result => {
        // 只在组件仍然挂载时更新状态
        if (isMounted.current) {
          setData(result);
        }
      });
  }, []);

  return data ? <div>{data.name}</div> : <div>加载中...</div>;
}
```

---

## useMemo 缓存计算结果

`useMemo` 缓存昂贵计算的结果，只在依赖变化时重新计算。

### 基本语法

```jsx
import { useMemo } from 'react';

const memoizedValue = useMemo(() => {
  return expensiveComputation(a, b);
}, [a, b]);
// 只有当 a 或 b 变化时，才会重新执行 expensiveComputation
```

### 缓存昂贵的计算

```jsx
function ProductList({ products, searchTerm, sortOrder }) {
  // ❌ 每次渲染都重新排序和过滤（可能有数万个产品）
  const filtered = products
    .filter(p => p.name.includes(searchTerm))
    .sort((a, b) => sortOrder === 'asc' ? a.price - b.price : b.price - a.price);

  // ✅ 只在 products、searchTerm 或 sortOrder 变化时重新计算
  const filtered = useMemo(() => {
    console.log('重新计算产品列表...');
    return products
      .filter(p => p.name.toLowerCase().includes(searchTerm.toLowerCase()))
      .sort((a, b) => sortOrder === 'asc' ? a.price - b.price : b.price - a.price);
  }, [products, searchTerm, sortOrder]);

  return (
    <ul>
      {filtered.map(p => (
        <li key={p.id}>{p.name} - ¥{p.price}</li>
      ))}
    </ul>
  );
}
```

### 统计数据计算

```jsx
function Dashboard({ transactions }) {
  const stats = useMemo(() => {
    const total = transactions.reduce((sum, t) => sum + t.amount, 0);
    const avg = total / transactions.length;
    const max = Math.max(...transactions.map(t => t.amount));
    const min = Math.min(...transactions.map(t => t.amount));

    return { total, avg, max, min, count: transactions.length };
  }, [transactions]);

  return (
    <div>
      <p>总计: ¥{stats.total.toFixed(2)}</p>
      <p>平均: ¥{stats.avg.toFixed(2)}</p>
      <p>最高: ¥{stats.max.toFixed(2)}</p>
      <p>最低: ¥{stats.min.toFixed(2)}</p>
      <p>笔数: {stats.count}</p>
    </div>
  );
}
```

---

## useMemo 保持引用稳定

当对象或数组作为 props 传递给子组件时，`useMemo` 可以保持引用稳定，避免不必要的子组件重渲染。

### 对象引用稳定

```jsx
function Parent() {
  const [name, setName] = useState('张三');
  const [theme, setTheme] = useState('dark');

  // ❌ 每次渲染都创建新对象，子组件总是重渲染
  // const style = { color: theme === 'dark' ? 'white' : 'black' };

  // ✅ 只有 theme 变化时才创建新对象
  const style = useMemo(
    () => ({ color: theme === 'dark' ? 'white' : 'black' }),
    [theme]
  );

  return (
    <div>
      <input value={name} onChange={e => setName(e.target.value)} />
      {/* name 变化时，style 引用不变，MemoizedChild 不会重渲染 */}
      <MemoizedChild style={style} />
    </div>
  );
}

const MemoizedChild = React.memo(function Child({ style }) {
  console.log('Child 渲染');
  return <div style={style}>子组件内容</div>;
});
```

### 数组引用稳定

```jsx
function FilterableList({ items }) {
  const [selectedIds, setSelectedIds] = useState([]);

  // ✅ 稳定引用，避免传递给子组件时导致不必要的重渲染
  const visibleItems = useMemo(
    () => items.filter(item => selectedIds.includes(item.id)),
    [items, selectedIds]
  );

  return <MemoizedItemList items={visibleItems} />;
}
```

---

## useCallback 缓存函数

`useCallback` 返回一个记忆化的回调函数，只在依赖变化时更新引用。

### 基本语法

```jsx
import { useCallback } from 'react';

const memoizedCallback = useCallback(() => {
  doSomething(a, b);
}, [a, b]);
// 等价于：
// useMemo(() => () => doSomething(a, b), [a, b])
```

### 配合 React.memo 使用

```jsx
function Parent() {
  const [count, setCount] = useState(0);
  const [name, setName] = useState('');

  // ❌ 每次渲染都创建新函数，React.memo 的浅比较失败
  // const handleClick = () => { console.log(count); };

  // ✅ 只有 count 变化时才更新函数引用
  const handleClick = useCallback(() => {
    console.log('当前计数:', count);
  }, [count]);

  return (
    <div>
      <input value={name} onChange={e => setName(e.target.value)} />
      <p>计数: {count}</p>
      <button onClick={() => setCount(c => c + 1)}>+1</button>
      {/* name 变化时，handleClick 引用不变，MemoizedButton 不会重渲染 */}
      <MemoizedButton onClick={handleClick}>记录计数</MemoizedButton>
    </div>
  );
}

const MemoizedButton = React.memo(function Button({ onClick, children }) {
  console.log('Button 渲染');
  return <button onClick={onClick}>{children}</button>;
});
```

### 传递给子组件的事件处理器

```jsx
function TodoApp() {
  const [todos, setTodos] = useState([]);
  const [filter, setFilter] = useState('all');

  const handleToggle = useCallback((id) => {
    setTodos(prev =>
      prev.map(todo =>
        todo.id === id ? { ...todo, done: !todo.done } : todo
      )
    );
  }, []);

  const handleDelete = useCallback((id) => {
    setTodos(prev => prev.filter(todo => todo.id !== id));
  }, []);

  // 使用函数式更新，所以不需要依赖 todos
  // useCallback 的依赖数组为空，引用始终稳定

  return (
    <div>
      <FilterBar filter={filter} onFilterChange={setFilter} />
      <TodoList
        todos={todos}
        filter={filter}
        onToggle={handleToggle}
        onDelete={handleDelete}
      />
    </div>
  );
}
```

---

## 何时不应该使用记忆化

### 过早优化是万恶之源

```jsx
// ❌ 不需要：计算本身很快
const result = useMemo(() => a + b, [a, b]);

// ✅ 直接计算
const result = a + b;

// ❌ 不需要：没有传递给 memo 包裹的子组件
const handler = useCallback(() => doSomething(), []);

// ✅ 直接定义
const handler = () => doSomething();
```

### 何时应该使用记忆化

| 场景 | 是否使用 |
|------|---------|
| 昂贵的计算（大数据排序/过滤/统计） | 使用 `useMemo` |
| 对象/数组作为 memo 子组件的 props | 使用 `useMemo` |
| 函数作为 memo 子组件的 props | 使用 `useCallback` |
| 作为其他 Hook 的依赖 | 视情况而定 |
| 简单的派生计算 | 不需要 |
| 事件处理器不传给 memo 子组件 | 不需要 |

### 记忆化有成本

`useMemo` 和 `useCallback` 本身也有开销：
- 需要比较依赖数组
- 需要保存上一次的结果
- 增加代码复杂度

**经验法则**：先不记忆化，只在发现性能问题时再添加。

---

## React.memo 包裹组件

`React.memo` 是一个高阶组件，当 props 没有变化时跳过重新渲染。

### 基本用法

```jsx
const MyComponent = React.memo(function MyComponent(props) {
  // 只有当 props 变化时才重新渲染
  return <div>{props.name}</div>;
});
```

### 浅比较的局限

```jsx
function Parent() {
  const [count, setCount] = useState(0);

  // ❌ 每次渲染创建新对象，浅比较认为 props 变了
  return <MemoizedChild style={{ color: 'red' }} />;

  // ✅ 使用 useMemo 稳定引用
  const style = useMemo(() => ({ color: 'red' }), []);
  return <MemoizedChild style={style} />;
}
```

### 自定义比较函数

```jsx
const areEqual = (prevProps, nextProps) => {
  // 返回 true 表示"相等，不需要重渲染"
  // 返回 false 表示"不相等，需要重渲染"
  return prevProps.user.id === nextProps.user.id;
};

const UserCard = React.memo(function UserCard({ user }) {
  return (
    <div>
      <p>{user.name}</p>
      <p>{user.email}</p>
    </div>
  );
}, areEqual);
```

### 完整的性能优化组合

```jsx
function App() {
  const [filter, setFilter] = useState('');
  const [items, setItems] = useState([]);

  // useMemo：缓存过滤结果
  const filteredItems = useMemo(
    () => items.filter(item => item.name.includes(filter)),
    [items, filter]
  );

  // useCallback：稳定的回调引用
  const handleItemClick = useCallback((id) => {
    console.log('点击了:', id);
  }, []);

  return (
    <div>
      <input value={filter} onChange={e => setFilter(e.target.value)} />
      {/* filteredItems 引用稳定 + handleItemClick 引用稳定
          → ItemList 只在 filteredItems 变化时重渲染 */}
      <ItemList items={filteredItems} onItemClick={handleItemClick} />
    </div>
  );
}

// React.memo：props 不变时不重渲染
const ItemList = React.memo(function ItemList({ items, onItemClick }) {
  console.log('ItemList 渲染');
  return (
    <ul>
      {items.map(item => (
        <MemoizedItem key={item.id} item={item} onClick={onItemClick} />
      ))}
    </ul>
  );
});

const MemoizedItem = React.memo(function Item({ item, onClick }) {
  return (
    <li onClick={() => onClick(item.id)}>
      {item.name}
    </li>
  );
});
```

---

## 依赖数组最佳实践

### useMemo 和 useCallback 的依赖规则

与 `useEffect` 相同：**列出函数体内使用的所有外部值**。

```jsx
function Search({ query, onSearch }) {
  // ✅ query 在函数体内使用，必须列入依赖
  const searchUrl = useMemo(
    () => `/api/search?q=${encodeURIComponent(query)}`,
    [query]
  );

  // ✅ onSearch 在函数体内使用，必须列入依赖
  const handleSubmit = useCallback(
    (e) => {
      e.preventDefault();
      onSearch(query);
    },
    [query, onSearch]
  );

  return <form onSubmit={handleSubmit}>...</form>;
}
```

### 使用函数式更新减少依赖

```jsx
function Counter() {
  const [count, setCount] = useState(0);

  // ❌ 依赖 count
  const increment = useCallback(() => {
    setCount(count + 1);
  }, [count]);

  // ✅ 不依赖 count，使用函数式更新
  const increment = useCallback(() => {
    setCount(prev => prev + 1);
  }, []);

  return <button onClick={increment}>计数: {count}</button>;
}
```

### ESLint 插件配置

确保 `eslint-plugin-react-hooks` 正确配置：

```json
{
  "rules": {
    "react-hooks/exhaustive-deps": "warn"
  }
}
```

> 不要轻易关闭 `exhaustive-deps` 警告。如果确实需要排除某个依赖，使用 `// eslint-disable-next-line` 并添加注释说明原因。

---

## 总结

| Hook | 用途 | 触发重新渲染 |
|------|------|-------------|
| `useRef` | 获取 DOM / 存储可变值 | 否 |
| `useMemo` | 缓存计算结果 / 稳定引用 | 否 |
| `useCallback` | 缓存函数引用 | 否 |
| `React.memo` | 缓存组件渲染 | 依赖 props 变化 |

**核心原则**：
1. 先写正确代码，再考虑优化
2. 使用 DevTools Profiler 定位真正的性能瓶颈
3. `useMemo`/`useCallback` 只在需要时使用，不要过度优化
4. 配合 `React.memo` 使用效果最佳
