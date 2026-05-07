# 自定义Hook类型

## 一、概念说明

自定义 Hook 是 React 中复用状态逻辑的核心机制，TypeScript 可以精确地推断或定义 Hook 的返回值类型。良好的类型设计让使用者无需关心内部实现，只通过类型就能了解 Hook 的接口。

**返回值类型**：使用 `as const` 元组或显式类型定义。

## 二、具体用法

### 2.1 基本自定义 Hook 类型

```tsx
import { useState, useEffect } from 'react';

// Hook 命名必须以 use 开头
// 返回值类型由 TypeScript 自动推断
function useCounter(initialValue = 0) {
  const [count, setCount] = useState(initialValue);

  const increment = () => setCount(c => c + 1);
  const decrement = () => setCount(c => c - 1);
  const reset = () => setCount(initialValue);

  // TypeScript 自动推断返回类型为
  // { count: number; increment: () => void; decrement: () => void; reset: () => void }
  return { count, increment, decrement, reset };
}

// 使用时类型完全推断
function Counter() {
  const { count, increment } = useCounter(10);
  // count: number, increment: () => void
  return <button onClick={increment}>{count}</button>;
}
```

### 2.2 返回元组的 Hook

```tsx
// 使用 as const 返回元组
function useToggle(initial = false) {
  const [value, setValue] = useState(initial);
  const toggle = () => setValue(v => !v);
  const setTrue = () => setValue(true);
  const setFalse = () => setValue(false);

  // as const 确保返回类型是 readonly 元组
  return [value, toggle, setTrue, setFalse] as const;
  // 类型: readonly [boolean, () => void, () => void, () => void]
}

// 解构使用
function Modal() {
  const [isOpen, toggleOpen, open, close] = useToggle(false);
  return (
    <div>
      <button onClick={open}>打开</button>
      {isOpen && <div>模态框内容</div>}
    </div>
  );
}
```

### 2.3 泛型自定义 Hook

```tsx
// 泛型 Hook — 适用于多种数据类型
function useLocalStorage<T>(key: string, initialValue: T) {
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch {
      return initialValue;
    }
  });

  const setValue = (value: T | ((val: T) => T)) => {
    const valueToStore = value instanceof Function ? value(storedValue) : value;
    setStoredValue(valueToStore);
    localStorage.setItem(key, JSON.stringify(valueToStore));
  };

  return [storedValue, setValue] as const;
  // 类型: readonly [T, (value: T | ((val: T) => T)) => void]
}

// 使用 — 类型自动推断
function App() {
  const [name, setName] = useLocalStorage('name', '');         // string
  const [count, setCount] = useLocalStorage('count', 0);       // number
  const [items, setItems] = useLocalStorage<string[]>('items', []); // 需显式指定
}
```

### 2.4 异步数据 Hook

```tsx
interface UseFetchResult<T> {
  data: T | null;
  loading: boolean;
  error: Error | null;
  refetch: () => void;
}

function useFetch<T>(url: string): UseFetchResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = async () => {
    setLoading(true);
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setData(json);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchData(); }, [url]);

  return { data, loading, error, refetch: fetchData };
}

// 使用 — 显式指定数据类型
interface User {
  id: number;
  name: string;
}

function UserProfile({ id }: { id: number }) {
  const { data, loading } = useFetch<User>(`/api/users/${id}`);
  if (loading) return <div>加载中...</div>;
  return <div>{data?.name}</div>;
}
```

## 三、注意事项与常见陷阱

1. **返回元组用 `as const`**：否则会被推断为数组联合类型而非元组
2. **泛型 Hook 在调用时指定类型**：`useLocalStorage<string[]>('key', [])`
3. **返回对象时 TypeScript 自动推断类型**：通常不需要显式标注
4. **Hook 的参数也应有类型**：复杂参数用 interface 定义
5. **遵循 Hook 规则**：自定义 Hook 内部可以调用其他 Hook
