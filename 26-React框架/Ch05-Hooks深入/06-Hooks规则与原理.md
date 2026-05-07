# Hooks 规则与原理

## 目录

1. [Hooks 的两条规则](#hooks-的两条规则)
2. [为什么有这些规则：链表实现](#为什么有这些规则链表实现)
3. [调用顺序为何至关重要](#调用顺序为何至关重要)
4. [条件/循环中调用的 Bug](#条件循环中调用的-bug)
5. [自定义 Hooks 的组合](#自定义-hooks-的组合)
6. [useSyncExternalStore](#usesyncexternalstore)
7. [useTransition 与 useDeferredValue](#usetransition-与-usedeferredvalue)
8. [useId](#useid)
9. [React Hooks 心智模型](#react-hooks-心智模型)

---

## Hooks 的两条规则

### 规则一：只在最顶层使用 Hook

不要在循环、条件判断或嵌套函数中调用 Hook。

```jsx
function Component({ shouldFetch }) {
  // ❌ 错误：条件中调用
  if (shouldFetch) {
    useEffect(() => { fetch('/api/data'); }, []);
  }

  // ✅ 正确：始终调用，在内部处理条件
  useEffect(() => {
    if (shouldFetch) {
      fetch('/api/data');
    }
  }, [shouldFetch]);
}
```

### 规则二：只在 React 函数中调用 Hook

只能在函数组件或自定义 Hook 中调用 Hook，不能在普通 JavaScript 函数或类组件中调用。

```jsx
// ✅ 函数组件
function MyComponent() {
  const [count, setCount] = useState(0);
}

// ✅ 自定义 Hook
function useCustomHook() {
  const [value, setValue] = useState('');
}

// ❌ 普通函数
function helperFunction() {
  useState(0); // 错误！
}
```

### ESLint 插件

```bash
npm install eslint-plugin-react-hooks --save-dev
```

```json
// .eslintrc
{
  "plugins": ["react-hooks"],
  "rules": {
    "react-hooks/rules-of-hooks": "error",
    "react-hooks/exhaustive-deps": "warn"
  }
}
```

---

## 为什么有这些规则：链表实现

React 内部使用**链表（Linked List）**来追踪组件中的 Hook 调用。每个组件有一个 hook 链表，每次调用 `useState`、`useEffect` 等 Hook 时，React 按照**调用顺序**在链表中创建或访问对应的节点。

### 数据结构示意

```jsx
// 组件代码
function Counter() {
  const [count, setCount] = useState(0);        // Hook 1
  const [name, setName] = useState('counter');   // Hook 2
  useEffect(() => { ... }, [count]);             // Hook 3
  const ref = useRef(null);                      // Hook 4
}

// React 内部的链表（简化表示）
{
  memoizedState: {
    hook: 'State',
    queue: { pending: null },
    memoizedState: 0,       // count = 0
    next: {                 // → Hook 2
      hook: 'State',
      queue: { pending: null },
      memoizedState: 'counter',  // name = 'counter'
      next: {               // → Hook 3
        hook: 'Effect',
        deps: [count],
        next: {             // → Hook 4
          hook: 'Ref',
          memoizedState: { current: null },
          next: null        // 链表结束
        }
      }
    }
  }
}
```

### 关键理解

React **不通过变量名**来识别 Hook，而是通过**调用顺序**。

```
第一次渲染: useState(0)       → 链表节点 1 (count)
           useState('counter') → 链表节点 2 (name)
           useEffect(...)      → 链表节点 3
           useRef(null)        → 链表节点 4

第二次渲染: useState(0)       → 访问链表节点 1 (count 的最新值)
           useState('counter') → 访问链表节点 2 (name 的最新值)
           useEffect(...)      → 访问链表节点 3
           useRef(null)        → 访问链表节点 4
```

如果调用顺序变了，React 会把 Hook "对错位"：

```jsx
// ❌ 这会导致链表错位
function Counter({ showExtra }) {
  const [count, setCount] = useState(0);  // Hook 1

  if (showExtra) {
    const [extra, setExtra] = useState(''); // Hook 2（条件性）
  }

  const [name, setName] = useState('');     // Hook 3?

  // 第一次渲染 (showExtra = true):
  //   链表: count → extra → name
  // 第二次渲染 (showExtra = false):
  //   链表: count → name
  //   React 认为 name 是原来的 extra！数据类型可能完全不同，直接崩溃
}
```

---

## 调用顺序为何至关重要

### 错误示例：条件中调用

```jsx
function Form({ isRegistered }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  if (isRegistered) {
    const [name, setName] = useState(''); // ❌ 条件性调用
  }

  const [age, setAge] = useState(0);

  // 当 isRegistered 从 true 变为 false:
  // 渲染 1: email(1) → password(2) → name(3) → age(4)  [4个节点]
  // 渲染 2: email(1) → password(2) → age(3)             [3个节点]
  //   React 认为 age 是原来的 name (节点3)!
  //   → useState 返回字符串而不是数字 → 崩溃
}
```

### 错误示例：循环中调用

```jsx
function List({ items }) {
  return items.map(item => {
    const [selected, setSelected] = useState(false); // ❌ 循环中调用
    return <li onClick={() => setSelected(!selected)}>{item.name}</li>;
  });

  // 如果 items 数量变化（比如从 3 个变为 2 个）:
  // 渲染 1: selected(1) → selected(2) → selected(3)  [3个节点]
  // 渲染 2: selected(1) → selected(2)                 [2个节点]
  //   第三个节点消失了，但 React 不知道哪个该消失
}
```

### 错误示例：提前 return

```jsx
function Component({ data }) {
  const [count, setCount] = useState(0);

  if (!data) return <Loading />; // ❌ 提前 return

  const [name, setName] = useState(data.name);

  // 渲染 1 (data = null):
  //   useState(0) → 执行 → return Loading (不再执行后面的 Hook)
  //   链表只有 1 个节点
  // 渲染 2 (data = { name: '张三' }):
  //   useState(0) → 链表节点 1
  //   useState(data.name) → 应该是链表节点 2，但只有 1 个节点!
  //   React 创建新节点 → 后续渲染链表长度不一致 → 崩溃
}
```

---

## 条件/循环中调用的 Bug

### 解决方案

**规则**：把条件/循环逻辑放在 Hook 内部，而不是放在 Hook 外部。

```jsx
// ❌ 杙误
if (isVisible) {
  useEffect(() => { ... });
}

// ✅ 正确
useEffect(() => {
  if (!isVisible) return;
  // ...
}, [isVisible]);
```

```jsx
// ❌ 错误
items.map(item => {
  const [state, setState] = useState(item.value);
});

// ✅ 正确：使用单独的子组件
function ItemList({ items }) {
  return items.map(item => <ItemRow key={item.id} item={item} />);
}

function ItemRow({ item }) {
  const [selected, setSelected] = useState(false);
  return (
    <li onClick={() => setSelected(!selected)}>{item.name}</li>
  );
}
```

### 有条件的需求

```jsx
function ChatRoom({ roomId, enableNotifications }) {
  // ✅ 始终调用所有 Hook
  const [messages, setMessages] = useState([]);
  const [isConnected, setIsConnected] = useState(false);

  // 通知效果：始终存在，在内部判断
  useEffect(() => {
    if (!enableNotifications) return;
    const handler = (msg) => showNotification(msg);
    messageBus.on('new', handler);
    return () => messageBus.off('new', handler);
  }, [enableNotifications]);

  // 连接效果：始终存在
  useEffect(() => {
    const conn = connect(roomId);
    conn.on('message', setMessages);
    setIsConnected(true);
    return () => {
      conn.disconnect();
      setIsConnected(false);
    };
  }, [roomId]);

  return <MessageList messages={messages} />;
}
```

---

## 自定义 Hooks 的组合

自定义 Hook 的调用也需要遵循同样的规则，因为最终它们会展开为普通 Hook 调用。

### 调用链展开

```jsx
function useAuth() {
  const [user, setUser] = useState(null);        // Hook A
  const [loading, setLoading] = useState(true);  // Hook B
  useEffect(() => { ... }, []);                   // Hook C
  return { user, loading, login, logout };
}

function usePermissions(user) {
  const [permissions, setPermissions] = useState([]); // Hook D
  useEffect(() => { ... }, [user]);                    // Hook E
  return permissions;
}

function Dashboard() {
  const { user, loading } = useAuth();           // 展开为 Hook A, B, C
  const permissions = usePermissions(user);       // 展开为 Hook D, E

  // React 内部看到的调用顺序:
  // useState(null)     ← Hook A (来自 useAuth)
  // useState(true)     ← Hook B (来自 useAuth)
  // useEffect(...)     ← Hook C (来自 useAuth)
  // useState([])       ← Hook D (来自 usePermissions)
  // useEffect(...)     ← Hook E (来自 usePermissions)
  // 这个顺序在每次渲染中必须一致！
}
```

### 错误的自定义 Hook 组合

```jsx
// ❌ 错误：自定义 Hook 中有条件调用
function useData(url) {
  if (!url) {
    return null; // 提前 return 跳过了 Hook 调用
  }

  const [data, setData] = useState(null);
  useEffect(() => { fetch(url).then(setData); }, [url]);
  return data;
}

// ✅ 正确
function useData(url) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!url) return; // 条件放在 effect 内部

    let cancelled = false;
    fetch(url)
      .then(res => res.json())
      .then(result => { if (!cancelled) setData(result); })
      .catch(err => { if (!cancelled) setError(err); });

    return () => { cancelled = true; };
  }, [url]);

  return { data, error };
}
```

---

## useSyncExternalStore

React 18 引入的 Hook，用于**安全地订阅外部数据源**，支持并发渲染模式。

### 为什么需要 useSyncExternalStore

在并发模式下，React 可能会多次调用渲染函数。如果直接读取外部 store，可能读到不一致的值：

```jsx
// ❌ 并发模式下的问题
function useExternalStore(store) {
  const [state, setState] = useState(store.getState());

  useEffect(() => {
    return store.subscribe(() => setState(store.getState()));
  }, [store]);

  return state;
  // 问题：并发模式下，store 可能在渲染期间变化
  // 导致屏幕上的 UI 与 store 中的值不一致（撕裂 / tearing）
}
```

### 基本用法

```jsx
import { useSyncExternalStore } from 'react';

function useOnlineStatus() {
  // subscribe: 订阅外部数据源的变化
  const isOnline = useSyncExternalStore(
    // 参数 1: subscribe（订阅函数）
    (callback) => {
      window.addEventListener('online', callback);
      window.addEventListener('offline', callback);
      return () => {
        window.removeEventListener('online', callback);
        window.removeEventListener('offline', callback);
      };
    },
    // 参数 2: getSnapshot（获取当前值的快照）
    () => navigator.onLine,
    // 参数 3: getServerSnapshot（服务端渲染时的值，可选）
    () => true
  );

  return isOnline;
}
```

### 订阅 Redux Store

```jsx
import { useSyncExternalStore } from 'react';

function useStoreValue(store, selector) {
  return useSyncExternalStore(
    store.subscribe,
    () => selector(store.getState())
  );
}

// 使用
function Counter() {
  const count = useStoreValue(store, state => state.count);
  return <p>计数: {count}</p>;
}
```

### 自定义 Store 示例

```jsx
// 创建一个简单的外部 store
function createCounterStore() {
  let state = { count: 0 };
  const listeners = new Set();

  return {
    getState: () => state,
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    increment: () => {
      state = { ...state, count: state.count + 1 };
      listeners.forEach(l => l());
    },
  };
}

const counterStore = createCounterStore();

// 在组件中使用
function Counter() {
  const count = useSyncExternalStore(
    counterStore.subscribe,
    () => counterStore.getState().count
  );

  return (
    <div>
      <p>计数: {count}</p>
      <button onClick={counterStore.increment}>+1</button>
    </div>
  );
}
```

---

## useTransition 与 useDeferredValue

React 18 引入的并发特性，用于标记非紧急的状态更新。

### useTransition

将某些更新标记为"过渡"（transitions），允许 React 在处理这些更新时保持页面的响应性。

```jsx
import { useState, useTransition } from 'react';

function SearchResults() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  // isPending: 是否有 transition 正在进行中
  // startTransition: 将回调内的状态更新标记为低优先级
  const [isPending, startTransition] = useTransition();

  const handleSearch = (value) => {
    // 紧急更新：立即更新输入框
    setQuery(value);

    // 非紧急更新：搜索结果的更新可以延迟
    startTransition(() => {
      // React 会优先响应用户的输入，然后在后台更新搜索结果
      const filtered = filterResults(value);
      setResults(filtered);
    });
  };

  return (
    <div>
      <input
        value={query}
        onChange={e => handleSearch(e.target.value)}
      />
      {/* 过渡进行中时显示加载状态，但输入框仍然可以交互 */}
      {isPending ? <Spinner /> : null}
      <ul>
        {results.map(r => <li key={r.id}>{r.name}</li>)}
      </ul>
    </div>
  );
}
```

### useTransition 的执行流程

```
用户输入 "React"
    │
    ▼
setInputValue("React") ← 紧急更新，立即生效
    │
    ▼
startTransition(() => { setFilteredResults(...) })
    │
    ▼
React 检查是否有更高优先级的更新
    │
    ├── 有（如用户又输入了）→ 先处理更高优先级的
    │
    └── 没有 → 执行 transition，更新搜索结果
    │
    ▼
isPending = false
```

### useDeferredValue

延迟某个值的更新，类似于自动的 debounce，但由 React 控制。

```jsx
import { useState, useDeferredValue, useMemo } from 'react';

function SearchPage() {
  const [query, setQuery] = useState('');
  // deferredQuery 的更新会延迟
  // 在 deferredQuery 更新期间，仍然显示旧的 deferredQuery 对应的 UI
  const deferredQuery = useDeferredValue(query);

  // 使用 deferredQuery 计算结果
  // 用户快速输入时，query 变化很快，但 deferredQuery 变化较慢
  // 旧的搜索结果会保持显示，直到 React 有空闲计算新的
  const results = useMemo(
    () => filterItems(deferredQuery),
    [deferredQuery]
  );

  // 通过比较判断当前显示的是否是"过时"数据
  const isStale = query !== deferredQuery;

  return (
    <div>
      <input value={query} onChange={e => setQuery(e.target.value)} />
      <div style={{ opacity: isStale ? 0.5 : 1 }}>
        <ResultsList results={results} />
      </div>
    </div>
  );
}
```

### useTransition vs useDeferredValue

| 特性 | useTransition | useDeferredValue |
|------|--------------|-----------------|
| 用途 | 标记状态更新为低优先级 | 延迟某个值的更新 |
| 控制方式 | 显式：调用 `startTransition` | 隐式：直接包裹值 |
| 加载状态 | 提供 `isPending` | 需手动比较判断 |
| 适用场景 | 已知哪些更新是低优先级的 | 只控制某个 prop 的更新时机 |

### 实际场景：大数据列表切换

```jsx
function TabbedList() {
  const [activeTab, setActiveTab] = useState('all');
  const [isPending, startTransition] = useTransition();

  const handleTabChange = (tab) => {
    // 紧急：立即高亮选中的 tab
    setActiveTab(tab);
  };

  // 数据量大，不需要延迟 tab 切换本身
  // 而是通过子组件使用 useDeferredValue

  return (
    <div>
      <TabBar active={activeTab} onChange={handleTabChange} />
      {isPending && <LoadingOverlay />}
      <HeavyList tab={activeTab} />
    </div>
  );
}

function HeavyList({ tab }) {
  const deferredTab = useDeferredValue(tab);
  const items = useMemo(() => computeItems(deferredTab), [deferredTab]);

  return items.map(item => <Item key={item.id} {...item} />);
}
```

---

## useId

React 18 引入，生成**唯一且稳定**的 ID，适用于服务端渲染（SSR）场景。

### 为什么不用简单计数器

```jsx
// ❌ 简单计数器在 SSR 下会有问题
let idCounter = 0;
function useIdBad() {
  return `id-${idCounter++}`;
}
// 服务端: id-0, id-1
// 客户端 hydration: 又从 id-0 开始 → 不匹配!
```

### 基本用法

```jsx
import { useId } from 'react';

function Form() {
  const nameId = useId();   // 例如: ":r0:"
  const emailId = useId();  // 例如: ":r1:"

  return (
    <form>
      <label htmlFor={nameId}>姓名</label>
      <input id={nameId} />

      <label htmlFor={emailId}>邮箱</label>
      <input id={emailId} type="email" />
    </form>
  );
}
```

### 与 aria 属性结合

```jsx
function PasswordInput({ label, error }) {
  const id = useId();
  const errorId = `${id}-error`;

  return (
    <div>
      <label htmlFor={id}>{label}</label>
      <input
        id={id}
        type="password"
        aria-describedby={error ? errorId : undefined}
        aria-invalid={!!error}
      />
      {error && (
        <p id={errorId} className="error">
          {error}
        </p>
      )}
    </div>
  );
}
```

### 在列表中使用

```jsx
function RadioGroup({ options, value, onChange }) {
  const groupId = useId();

  return (
    <div role="radiogroup">
      {options.map((option, index) => {
        const optionId = `${groupId}-${index}`;
        return (
          <div key={option.value}>
            <input
              type="radio"
              id={optionId}
              name={groupId}
              value={option.value}
              checked={value === option.value}
              onChange={() => onChange(option.value)}
            />
            <label htmlFor={optionId}>{option.label}</label>
          </div>
        );
      })}
    </div>
  );
}
```

---

## React Hooks 心智模型

### 函数组件 = 状态 + 渲染 + 副作用

```
┌─────────────────────────────────────┐
│           函数组件                    │
│                                     │
│  ┌──────────┐  ┌────────────────┐   │
│  │ 状态 Hooks │  │ 副作用 Hooks   │   │
│  │          │  │                │   │
│  │ useState │  │ useEffect      │   │
│  │ useReducer│  │ useLayoutEffect│   │
│  │ useRef   │  │ useCallback    │   │
│  │ useMemo  │  │                │   │
│  └──────────┘  └────────────────┘   │
│                                     │
│  每个 Hook 独立运作，通过调用顺序      │
│  在链表中关联                         │
└─────────────────────────────────────┘
```

### Hooks 的本质

每个 Hook 在 React 内部是一个"挂载"（mount）或"更新"（update）的函数：

```jsx
// React 内部（简化）
function mountState(initialState) {
  const hook = mountWorkInProgressHook();

  hook.memoizedState = hook.baseState = initialState;
  const queue = hook.queue = { pending: null };

  const dispatch = (action) => {
    // 计算新状态
    const newState = typeof action === 'function'
      ? action(hook.baseState)
      : action;

    hook.baseState = newState;
    hook.queue.pending = /* 更新队列 */;
    scheduleUpdateOnFiber(); // 调度重新渲染
  };

  return [hook.memoizedState, dispatch];
}

function updateState() {
  const hook = updateWorkInProgressHook(); // 按顺序获取链表中的下一个 hook
  // ... 处理 pending 更新队列
  return [hook.memoizedState, hook.queue.dispatch];
}
```

### 闭包与 Hooks

Hooks 大量利用闭包来捕获和隔离状态：

```jsx
function Counter() {
  const [count, setCount] = useState(0);

  useEffect(() => {
    // 这个 effect 捕获了当前渲染时的 count 值（闭包）
    const timer = setInterval(() => {
      console.log(count); // 固定为 0（如果是首次渲染且依赖为空）
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  // 每次渲染都会创建新的 effect 函数
  // 每个 effect 函数捕获的是该次渲染的 count 值
  // cleanup 清除的是上一次 effect 的闭包
}
```

### 渲染与 Hooks 的关系

```
组件首次渲染:
  执行函数体 → 调用 Hooks → React 创建链表 → 提交 DOM → 执行 effects

组件更新（setState 触发）:
  重新执行函数体 → 按顺序复用链表中的 Hook 节点 → 更新 DOM → cleanup 上一个 effects → 执行新的 effects

组件卸载:
  执行所有 effect 的 cleanup 函数 → 释放链表
```

### 理解 stale closure 的根本原因

```jsx
function Counter() {
  const [count, setCount] = useState(0);

  // 这次渲染时，函数体中的 count 是 0
  // 这个值被闭包捕获在 effect 函数中
  useEffect(() => {
    console.log('effect 中的 count:', count); // 0
  }, [count]);

  // 如果 effect 的依赖数组为空，则永远捕获的是 count = 0
  // 这就是 stale closure：闭包捕获了过时的值

  return <button onClick={() => setCount(c => c + 1)}>计数</button>;
}
```

---

## React 18+ 新增 Hooks 概览

| Hook | 用途 | 版本 |
|------|------|------|
| `useSyncExternalStore` | 安全订阅外部数据源 | 18.0 |
| `useTransition` | 标记低优先级更新 | 18.0 |
| `useDeferredValue` | 延迟值更新 | 18.0 |
| `useId` | 生成 SSR 安全的唯一 ID | 18.0 |
| `use` | 读取 Promise/Context（实验性） | 19.0 |
| `useOptimistic` | 乐观更新 UI（实验性） | 19.0 |
| `useActionState` | 表单 Action 状态管理（实验性） | 19.0 |

---

## 总结

1. **规则的根源**是 React 使用**链表**追踪 Hook，调用顺序必须一致
2. **ESLint 插件**会自动检查这些规则，不要轻易关闭
3. **条件/循环逻辑**放在 Hook 内部，而不是包裹 Hook
4. **并发模式**引入了新的 Hooks：`useTransition`、`useDeferredValue`、`useSyncExternalStore`
5. **useId** 解决了 SSR 中 ID 不一致的问题
6. **心智模型**：每次渲染都是独立的函数调用，Hooks 通过链表连接这些独立的渲染结果
