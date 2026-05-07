# useRef 详解

`useRef` 是 React 提供的 Hook，用于创建一个可变的引用对象，该对象在组件的整个生命周期中保持不变。

---

## 一、基础语法

### 1.1 useRef 基本结构

```jsx
import { useRef } from 'react';

function MyComponent() {
  const ref = useRef(initialValue);

  // ref 是一个对象：{ current: initialValue }
  console.log(ref.current);  // 访问当前值
  ref.current = newValue;    // 修改值（不触发重渲染）

  return <div />;
}
```

### 1.2 ref 对象的特性

```jsx
function Demo() {
  const ref = useRef(0);

  console.log('组件渲染了');
  console.log('ref.current:', ref.current);

  return (
    <div>
      <p>ref 值: {ref.current}</p>
      <button onClick={() => {
        ref.current += 1;
        console.log('修改后:', ref.current);
        // 注意：修改 ref.current 不会触发重渲染！
        // 页面上的数字不会更新
      }}>
        增加 ref
      </button>
      <button onClick={() => console.log('当前值:', ref.current)}>
        打印当前值
      </button>
    </div>
  );
}
```

**关键特性**：
- 修改 `ref.current` **不会触发组件重渲染**
- `ref` 对象在组件的每次渲染中都是**同一个引用**
- `ref.current` 是可变的，可以直接赋值

---

## 二、ref vs state

| 特性 | useRef | useState |
|---|---|---|
| 修改后重渲染 | 否 | 是 |
| 值的读取 | `ref.current` | 直接使用 state 变量 |
| 更新方式 | 直接赋值 `ref.current = x` | 调用 setState(x) |
| 更新时机 | 同步，立即生效 | 异步（批处理后生效） |
| 使用场景 | 存储不需要触发 UI 更新的值 | 需要触发 UI 更新的值 |

### 示例对比

```jsx
function Timer() {
  const [count, setCount] = useState(0);     // 需要显示在 UI 上
  const intervalRef = useRef(null);            // 定时器 ID，不需要显示

  const start = () => {
    // 存储 interval ID，不需要触发渲染
    intervalRef.current = setInterval(() => {
      setCount(c => c + 1);  // count 需要更新 UI
    }, 1000);
  };

  const stop = () => {
    clearInterval(intervalRef.current);  // 使用 ref 中存储的 ID
  };

  return (
    <div>
      <p>{count}</p>
      <button onClick={start}>开始</button>
      <button onClick={stop}>停止</button>
    </div>
  );
}
```

---

## 三、ref 作为可变容器

### 3.1 存储定时器 ID

```jsx
function Stopwatch() {
  const [time, setTime] = useState(0);
  const [running, setRunning] = useState(false);
  const intervalRef = useRef(null);

  useEffect(() => {
    if (running) {
      intervalRef.current = setInterval(() => {
        setTime(t => t + 10);  // 每 10ms 更新
      }, 10);
    } else {
      clearInterval(intervalRef.current);
    }

    return () => clearInterval(intervalRef.current);  // 清理
  }, [running]);

  return (
    <div>
      <span>{(time / 1000).toFixed(2)}s</span>
      <button onClick={() => setRunning(!running)}>
        {running ? '暂停' : '开始'}
      </button>
      <button onClick={() => { setRunning(false); setTime(0); }}>
        重置
      </button>
    </div>
  );
}
```

### 3.2 存储上一次的值

```jsx
function usePrevious(value) {
  const ref = useRef();

  useEffect(() => {
    ref.current = value;  // 渲染完成后更新 ref
  });

  return ref.current;  // 返回上一次的值
}

// 使用
function Counter() {
  const [count, setCount] = useState(0);
  const prevCount = usePrevious(count);

  return (
    <div>
      <p>当前: {count}, 上一次: {prevCount}</p>
      <button onClick={() => setCount(c => c + 1)}>+1</button>
    </div>
  );
}
```

### 3.3 缓存计算结果

```jsx
function ExpensiveComponent({ data }) {
  const cachedRef = useRef({ key: null, result: null });

  const result = (() => {
    // 如果数据没变，返回缓存结果
    if (cachedRef.current.key === data) {
      return cachedRef.current.result;
    }
    // 否则重新计算
    const computed = expensiveComputation(data);
    cachedRef.current = { key: data, result: computed };
    return computed;
  })();

  return <div>{result}</div>;
}
```

### 3.4 存储最新值（避免闭包陷阱）

```jsx
function useLatest(value) {
  const ref = useRef(value);

  // 每次渲染后更新为最新值
  useEffect(() => {
    ref.current = value;
  });

  return ref;
}

// 使用：避免 useEffect 中的闭包陈旧值
function ChatRoom({ onMessage }) {
  const onMessageRef = useLatest(onMessage);

  useEffect(() => {
    const socket = new WebSocket('wss://chat.example.com');

    socket.onmessage = (event) => {
      // 使用 ref.current 获取最新的 onMessage
      // 不会导致 effect 依赖 onMessage 而频繁重新连接
      onMessageRef.current(event.data);
    };

    return () => socket.close();
  }, []);  // 空依赖：只连接一次

  return <div>聊天室</div>;
}
```

---

## 四、useRef(null) 模式

### 4.1 获取 DOM 引用

最常见的用法：获取对 DOM 元素的引用。

```jsx
function TextInput() {
  const inputRef = useRef(null);

  const handleClick = () => {
    // inputRef.current 就是实际的 DOM 元素
    inputRef.current.focus();
  };

  return (
    <div>
      <input ref={inputRef} type="text" />
      <button onClick={handleClick}>聚焦</button>
    </div>
  );
}
```

### 4.2 初始值为 null 的原因

```jsx
// 组件首次渲染时 DOM 还不存在
// 所以 ref.current 初始值为 null
// 渲染完成后 React 会将 DOM 节点赋值给 ref.current

function Demo() {
  const divRef = useRef(null);

  useEffect(() => {
    // 此时 DOM 已经创建完成
    console.log(divRef.current);  // <div>...</div>
    console.log(divRef.current.getBoundingClientRect());
  }, []);

  // 首次渲染时 divRef.current 为 null
  // 但这个 null 不影响 JSX 的渲染
  return <div ref={divRef}>内容</div>;
}
```

---

## 五、Callback Ref

除了 `useRef`，还可以使用回调 ref，每次 ref 变化时调用回调函数。

### 5.1 基础用法

```jsx
function TextInput() {
  const [inputElement, setInputElement] = useState(null);

  const inputRef = useCallback((node) => {
    // node 是 DOM 元素或 null（卸载时）
    if (node !== null) {
      setInputElement(node);
      node.focus();
    }
  }, []);

  return <input ref={inputRef} type="text" />;
}
```

### 5.2 useRef vs Callback Ref

| 特性 | useRef | callback ref |
|---|---|---|
| 获取 DOM | 需要 useEffect | 回调中直接获取 |
| 设置时机 | 渲染后异步获取 | 渲染过程中同步调用 |
| 动态元素 | 需要额外处理 | 自动处理 |
| 多个元素 | 多个 ref 对象 | 可以动态管理 |
| 常见用途 | 简单 DOM 引用 | 动画库集成、复杂交互 |

### 5.3 动态列表中的 Callback Ref

```jsx
function DynamicList({ items }) {
  const [itemElements, setItemElements] = useState({});

  const getItemRef = useCallback((id) => (node) => {
    if (node) {
      setItemElements(prev => ({ ...prev, [id]: node }));
    } else {
      setItemElements(prev => {
        const next = { ...prev };
        delete next[id];
        return next;
      });
    }
  }, []);

  // 使用 itemElements 进行动画等操作
  useEffect(() => {
    Object.values(itemElements).forEach(el => {
      // 对每个元素执行操作
    });
  }, [itemElements]);

  return (
    <ul>
      {items.map(item => (
        <li key={item.id} ref={getItemRef(item.id)}>
          {item.text}
        </li>
      ))}
    </ul>
  );
}
```

---

## 六、ref 在 React 生命周期中的时机

```
组件挂载
  │
  ├── 1. 函数组件执行（渲染）
  │      ref.current = null（首次）
  │
  ├── 2. React 创建 DOM
  │
  ├── 3. React 将 DOM 赋值给 ref.current
  │
  ├── 4. useEffect 执行
  │      此时 ref.current 已经可用
  │
  ├── 5. useLayoutEffect 执行
  │      此时 ref.current 已经可用（更早）
  │
  ▼
组件更新
  │
  ├── 1. 函数组件执行（重新渲染）
  │      ref.current 保持不变
  │
  ├── 2. React 更新 DOM
  │
  ├── 3. 如果 ref 关联的 DOM 变化，更新 ref.current
  │
  ├── 4. useEffect 执行
  │
  ▼
组件卸载
  │
  ├── 1. ref.current = null
  │
  └── 2. cleanup 函数执行（useEffect 的 return）
```

---

## 七、常见陷阱

### 陷阱 1：在渲染期间读取 ref

```jsx
function Bad() {
  const ref = useRef(null);

  // ❌ 错误：渲染期间 ref.current 可能为 null
  const width = ref.current?.getBoundingClientRect().width || 0;

  return <div ref={ref}>宽度: {width}</div>;
  // 首次渲染时 ref.current 为 null，width 为 0
}

function Good() {
  const ref = useRef(null);
  const [width, setWidth] = useState(0);

  // ✅ 正确：在 effect 中读取
  useLayoutEffect(() => {
    setWidth(ref.current.getBoundingClientRect().width);
  }, []);

  return <div ref={ref}>宽度: {width}</div>;
}
```

### 陷阱 2：修改 ref 不触发重渲染

```jsx
function Bad() {
  const countRef = useRef(0);

  return (
    <div>
      <span>{countRef.current}</span>
      <button onClick={() => { countRef.current += 1; }}>
        +1
      </button>
      {/* 点击后页面数字不会变！ref 变化不触发重渲染 */}
    </div>
  );
}

function Good() {
  const [count, setCount] = useState(0);
  const countRef = useRef(count);

  useEffect(() => { countRef.current = count; }, [count]);

  return (
    <div>
      <span>{count}</span>
      <button onClick={() => setCount(c => c + 1)}>+1</button>
    </div>
  );
}
```

### 陷阱 3：将 ref 作为 useEffect 依赖

```jsx
function Bad() {
  const ref = useRef(null);

  useEffect(() => {
    const el = ref.current;
    // 某些操作...
  }, [ref]);  // ❌ ref 对象引用永远不变，这个依赖没有意义
}

function Good() {
  const ref = useRef(null);

  useEffect(() => {
    const el = ref.current;
    // 某些操作...
  }, []);  // ✅ 空依赖，因为 ref 本身不变
  // 或者使用 ref.current 作为依赖（如果需要检测 DOM 变化）
}
```
