# useContext类型

## 一、概念说明

`useContext` 用于在组件树中共享数据，TypeScript 通过泛型为 Context 提供类型安全。核心挑战是处理 Context 值为 `null`（未提供 Provider）的情况，以及区分"读"和"写"操作的类型。

## 二、具体用法

### 2.1 基本 Context 类型

```tsx
import { createContext, useContext, useState } from 'react';

// 定义 Context 值的类型
interface ThemeContextType {
  theme: 'light' | 'dark';
  toggleTheme: () => void;
}

// 创建 Context，提供默认值
const ThemeContext = createContext<ThemeContextType>({
  theme: 'light',
  toggleTheme: () => {},
});

// Provider 组件
function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<'light' | 'dark'>('light');

  const toggleTheme = () => {
    setTheme(prev => (prev === 'light' ? 'dark' : 'light'));
  };

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

// 消费组件
function ThemedButton() {
  const { theme, toggleTheme } = useContext(ThemeContext);
  return (
    <button onClick={toggleTheme} className={`btn-${theme}`}>
      当前主题: {theme}
    </button>
  );
}
```

### 2.2 处理 Context 为 null 的情况

```tsx
// 方式一：初始值为 null，使用自定义 Hook 检查
interface AuthContextType {
  user: { name: string; role: string } | null;
  login: (username: string) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

// 自定义 Hook 封装 null 检查
function useAuth(): AuthContextType {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth 必须在 AuthProvider 内使用');
  }
  return context;
}

// 使用 — 不需要再做 null 检查
function UserProfile() {
  const { user, logout } = useAuth(); // 类型安全，肯定不是 null
  return <div>{user?.name}</div>;
}
```

### 2.3 分离读写 Context

```tsx
// 将 Context 分为只读和只写，优化性能
interface CountContextType {
  count: number;
  increment: () => void;
}

// 只读 Context
const CountStateContext = createContext<number>(0);
// 只写 Context
const CountActionsContext = createContext<{ increment: () => void }>({
  increment: () => {},
});

function CountProvider({ children }: { children: React.ReactNode }) {
  const [count, setCount] = useState(0);
  const increment = () => setCount(c => c + 1);

  return (
    <CountActionsContext.Provider value={{ increment }}>
      <CountStateContext.Provider value={count}>
        {children}
      </CountStateContext.Provider>
    </CountActionsContext.Provider>
  );
}

// 子组件只订阅需要的部分
function CounterDisplay() {
  const count = useContext(CountStateContext); // 只有 count 变化时重新渲染
  return <span>{count}</span>;
}

function CounterButton() {
  const { increment } = useContext(CountActionsContext);
  return <button onClick={increment}>+1</button>;
}
```

### 2.4 泛型 Context Hook

```tsx
// 创建 Context 的工厂函数
function createTypedContext<T>() {
  const Context = createContext<T | undefined>(undefined);

  function useValue(): T {
    const context = useContext(Context);
    if (context === undefined) {
      throw new Error('useValue 必须在 Provider 内使用');
    }
    return context;
  }

  return [Context.Provider, useValue] as const;
}

// 使用
const [SettingsProvider, useSettings] = createTypedContext<{
  locale: string;
  timezone: string;
}>();
```

## 三、注意事项与常见陷阱

1. **不要在初始值中放空对象**：`createContext({} as AuthContextType)` 会导致运行时错误
2. **始终创建自定义 `useContext` 封装**：统一处理 null 检查逻辑
3. **Context 值变化会导致所有消费者重新渲染**：拆分读写 Context 优化性能
4. **Provider 的 `value` 对象应该被 memo**：避免不必要的重新渲染
5. **泛型 Context 是类型安全的工厂模式**：推荐在大型项目中使用
