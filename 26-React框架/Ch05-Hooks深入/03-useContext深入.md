# useContext 深入

## 目录

1. [基础用法](#基础用法)
2. [Context 与默认值](#context-与默认值)
3. [多个 Context](#多个-context)
4. [Context 嵌套](#context-嵌套)
5. [性能考量](#性能考量)
6. [Context Selector 模式](#context-selector-模式)
7. [Context vs Prop Drilling](#context-vs-prop-drilling)

---

## 基础用法

Context 提供了一种在组件树中共享数据的方式，无需逐层手动传递 props。

### 三个步骤

```jsx
import { createContext, useContext, useState } from 'react';

// 步骤 1：创建 Context
const ThemeContext = createContext();

// 步骤 2：在顶层组件中提供值
function App() {
  const [theme, setTheme] = useState('light');

  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      <Toolbar />
    </ThemeContext.Provider>
  );
}

// 步骤 3：在任意后代组件中消费
function ThemedButton() {
  const { theme, setTheme } = useContext(ThemeContext);

  return (
    <button
      className={`btn-${theme}`}
      onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}
    >
      当前主题: {theme}
    </button>
  );
}
```

### Context 的数据流

```
App
 │  value={{ theme, setTheme }}
 ▼
ThemeContext.Provider
 │
 ├── Toolbar
 │    ├── Menu
 │    │    └── ThemedButton ← useContext(ThemeContext) 可以在这里消费
 │    └── Header ← 也可以在这里消费
 └── Footer ← 也可以在这里消费

任何后代组件都可以直接访问，无需中间组件传递。
```

---

## Context 与默认值

### 默认值的触发条件

`createContext(defaultValue)` 中的默认值**仅在组件没有找到 Provider 时使用**：

```jsx
const AuthContext = createContext({
  user: null,
  isLoggedIn: false,
  login: () => {},
  logout: () => {},
});

// 如果没有被 Provider 包裹，useContext 返回默认值
function Demo() {
  const auth = useContext(AuthContext);
  // auth = { user: null, isLoggedIn: false, login: ..., logout: ... }
  // 这是默认值，不是 Provider 提供的值
}
```

### 利用默认值进行单元测试

```jsx
// 测试时不需要 Provider 包裹
test('未登录时显示登录按钮', () => {
  render(<Navbar />);
  expect(screen.getByText('登录')).toBeInTheDocument();
});
```

### 使用 undefined 作为默认值

```jsx
// 当 context 值必须由 Provider 提供时
const UserContext = createContext(undefined);

function useUser() {
  const context = useContext(UserContext);
  if (context === undefined) {
    throw new Error('useUser 必须在 UserProvider 内使用');
  }
  return context;
}
```

---

## 多个 Context

组件可以同时消费多个 Context：

```jsx
const ThemeContext = createContext('light');
const AuthContext = createContext(null);
const LocaleContext = createContext('zh-CN');

function App() {
  return (
    <ThemeContext.Provider value="dark">
      <AuthContext.Provider value={{ user: '张三' }}>
        <LocaleContext.Provider value="zh-CN">
          <Dashboard />
        </LocaleContext.Provider>
      </AuthContext.Provider>
    </ThemeContext.Provider>
  );
}

function Dashboard() {
  const theme = useContext(ThemeContext);
  const { user } = useContext(AuthContext);
  const locale = useContext(LocaleContext);

  return (
    <div className={`theme-${theme}`}>
      <p>欢迎, {user}</p>
      <p>语言: {locale}</p>
    </div>
  );
}
```

### 封装自定义 Hook 简化多 Context 使用

```jsx
function useAppContext() {
  const theme = useContext(ThemeContext);
  const auth = useContext(AuthContext);
  const locale = useContext(LocaleContext);

  return { theme, auth, locale };
}

// 使用
function Dashboard() {
  const { theme, auth, locale } = useAppContext();
  return <div>...</div>;
}
```

---

## Context 嵌套

### 同一 Context 的多层 Provider

嵌套的同类型 Provider 会覆盖外层的值，最近的 Provider 生效：

```jsx
const ThemeContext = createContext('light');

function App() {
  return (
    <ThemeContext.Provider value="dark">
      <Toolbar />           {/* theme = 'dark' */}
      <ThemeContext.Provider value="light">
        <Sidebar />         {/* theme = 'light' (被内层覆盖) */}
      </ThemeContext.Provider>
    </ThemeContext.Provider>
  );
}

function Toolbar() {
  const theme = useContext(ThemeContext);
  return <div className={theme}>工具栏（暗色主题）</div>;
}

function Sidebar() {
  const theme = useContext(ThemeContext);
  return <div className={theme}>侧边栏（亮色主题）</div>;
}
```

### 实际场景：局部覆盖

```jsx
function App() {
  return (
    <ThemeProvider theme="corporate">
      <Header />
      <main>
        {/* 特定区域使用不同主题 */}
        <ThemeProvider theme="creative">
          <DesignEditor />
        </ThemeProvider>
        <AdminPanel /> {/* 仍然是 corporate 主题 */}
      </main>
    </ThemeProvider>
  );
}
```

---

## 性能考量

### 问题：Context 的值变化会导致所有消费者重新渲染

```jsx
const AppContext = createContext();

function App() {
  const [user, setUser] = useState(null);
  const [theme, setTheme] = useState('light');

  // ❌ 问题：每次 App 渲染，{ user, theme } 是新对象
  // 导致所有消费者即使不需要也会重新渲染
  return (
    <AppContext.Provider value={{ user, theme }}>
      <ExpensiveComponent /> {/* user 变了就会重渲染，即使它只关心 theme */}
    </AppContext.Provider>
  );
}
```

### 解决方案一：拆分 Context

```jsx
// ✅ 将不同关注点拆分到不同 Context
const UserContext = createContext();
const ThemeContext = createContext();

function App() {
  const [user, setUser] = useState(null);
  const [theme, setTheme] = useState('light');

  return (
    <UserContext.Provider value={user}>
      <ThemeContext.Provider value={theme}>
        <UserDisplay />    {/* 只在 user 变化时重渲染 */}
        <ThemedComponent /> {/* 只在 theme 变化时重渲染 */}
      </ThemeContext.Provider>
    </UserContext.Provider>
  );
}
```

### 解决方案二：稳定 Context 值的引用

```jsx
function App() {
  const [user, setUser] = useState(null);
  const [theme, setTheme] = useState('light');

  // ✅ 使用 useMemo 稳定引用，只在真正需要时创建新对象
  const contextValue = useMemo(
    () => ({ user, theme, setUser, setTheme }),
    [user, theme]
  );

  return (
    <AppContext.Provider value={contextValue}>
      <Consumers />
    </AppContext.Provider>
  );
}
```

### 解决方案三：Context + useReducer

```jsx
const AppContext = createContext();

function appReducer(state, action) {
  switch (action.type) {
    case 'SET_USER': return { ...state, user: action.payload };
    case 'SET_THEME': return { ...state, theme: action.payload };
    default: return state;
  }
}

function AppProvider({ children }) {
  const [state, dispatch] = useReducer(appReducer, {
    user: null,
    theme: 'light',
  });

  // ✅ dispatch 引用始终稳定，state 变化时才重渲染
  const value = useMemo(() => ({ state, dispatch }), [state]);

  return (
    <AppContext.Provider value={value}>
      {children}
    </AppContext.Provider>
  );
}
```

### 使用 React.memo 防止中间组件重渲染

```jsx
// 即使 Context 值变化，用 React.memo 包裹的中间组件
// 如果 props 没变，也不会重渲染
const MiddleComponent = React.memo(function MiddleComponent({ children }) {
  console.log('MiddleComponent 渲染'); // Context 变化时不会输出
  return <div>{children}</div>;
});

function App() {
  return (
    <ThemeContext.Provider value={theme}>
      <MiddleComponent>
        <ThemedButton /> {/* 只有这个消费组件会重渲染 */}
      </MiddleComponent>
    </ThemeContext.Provider>
  );
}
```

---

## Context Selector 模式

React 原生不支持 context selector（类似 Redux 的 `useSelector`），但有社区方案。

### 问题：只能消费整个 Context 值

```jsx
// 原生 useContext 只能获取整个值
const { user, theme, locale, notifications } = useContext(AppContext);
// 即使只需要 user，theme/locale/notifications 变化也会导致重渲染
```

### 方案一：useContextSelector（社区库）

```bash
npm install use-context-selector
```

```jsx
import { createContext, useContextSelector } from 'use-context-selector';

const AppContext = createContext();

function App() {
  const [state, setState] = useState({ user: null, theme: 'light' });

  return (
    <AppContext.Provider value={state}>
      <UserDisplay />
    </AppContext.Provider>
  );
}

// ✅ 只订阅 user，theme 变化不会导致重渲染
function UserDisplay() {
  const user = useContextSelector(AppContext, v => v.user);
  return <p>{user?.name}</p>;
}
```

### 方案二：拆分 Context + 自定义 Provider

```jsx
// ✅ 纯 React 方案：通过子组件自动订阅自己需要的 context
const StateContext = createContext();
const DispatchContext = createContext();

function AppProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState);

  return (
    <StateContext.Provider value={state}>
      <DispatchContext.Provider value={dispatch}>
        {children}
      </DispatchContext.Provider>
    </StateContext.Provider>
  );
}

// dispatch 引用稳定，使用 dispatch 的组件不会因 state 变化而重渲染
function AddTodoButton() {
  const dispatch = useContext(DispatchContext);
  // dispatch 永远是同一个引用，state 变化不会导致此组件重渲染
  return <button onClick={() => dispatch({ type: 'ADD_TODO' })}>添加</button>;
}

// 只有需要读取 state 的组件才会在 state 变化时重渲染
function TodoCount() {
  const state = useContext(StateContext);
  return <p>共 {state.todos.length} 项</p>;
}
```

---

## Context vs Prop Drilling

### Prop Drilling 的问题

```jsx
// ❌ Prop Drilling：中间组件被迫传递不需要的 props
function App() {
  const [user, setUser] = useState(null);
  return <Layout user={user} setUser={setUser} />;
}

function Layout({ user, setUser }) {
  // Layout 不需要 user，但必须传递
  return (
    <div>
      <Header user={user} setUser={setUser} />
      <Main />
    </div>
  );
}

function Header({ user, setUser }) {
  // Header 也不直接需要 user
  return <UserMenu user={user} setUser={setUser} />;
}

function UserMenu({ user, setUser }) {
  // 最终使用者
  return <span>{user?.name}</span>;
}
```

### Context 的解决方案

```jsx
// ✅ Context：直接在需要的地方消费
function App() {
  const [user, setUser] = useState(null);
  return (
    <UserContext.Provider value={{ user, setUser }}>
      <Layout />
    </UserContext.Provider>
  );
}

function Layout() {
  return (
    <div>
      <Header />
      <Main />
    </div>
  );
}

function Header() {
  return <UserMenu />;
}

function UserMenu() {
  const { user, setUser } = useContext(UserContext);
  return <span>{user?.name}</span>;
}
```

### 选择指南

| 场景 | 推荐方案 |
|------|---------|
| 主题、语言、认证信息等全局共享数据 | Context |
| 只有少数几层传递 | Prop Drilling（更明确） |
| 组件需要在不同树中复用 | Prop Drilling（保持独立性） |
| 数据变化频繁 | 考虑状态管理库（Zustand、Jotai） |
| 需要精细控制渲染 | 拆分 Context 或使用状态管理库 |
| 需要派生/计算状态 | 状态管理库（支持 selector） |

### 何时不应该用 Context

```jsx
// ❌ 不要用于高频更新的状态（如动画帧、鼠标位置）
const MouseContext = createContext();

function MouseTracker() {
  const [pos, setPos] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handler = (e) => setPos({ x: e.clientX, y: e.clientY });
    window.addEventListener('mousemove', handler);
    return () => window.removeEventListener('mousemove', handler);
  }, []);

  // ❌ 每秒可能更新 60+ 次，所有消费者都会重渲染
  return (
    <MouseContext.Provider value={pos}>
      <App />
    </MouseContext.Provider>
  );
}

// ✅ 高频更新状态用 ref + 手动 DOM 操作，或状态管理库
function CursorFollower() {
  const dotRef = useRef(null);

  useEffect(() => {
    const handler = (e) => {
      if (dotRef.current) {
        dotRef.current.style.transform =
          `translate(${e.clientX}px, ${e.clientY}px)`;
      }
    };
    window.addEventListener('mousemove', handler);
    return () => window.removeEventListener('mousemove', handler);
  }, []);

  return <div ref={dotRef} className="cursor-dot" />;
}
```

---

## 最佳实践总结

1. **按关注点拆分 Context**：不要把所有全局状态放在一个 Context 里
2. **稳定 Provider 的 value**：使用 `useMemo` 避免不必要的重渲染
3. **提供有意义的默认值**：方便测试和独立组件开发
4. **封装自定义 Hook**：`useTheme()`、`useAuth()` 比直接用 `useContext` 更清晰
5. **不要用于高频更新**：鼠标位置、动画状态等不适合用 Context
6. **考虑组合方案**：Context + useReducer 适合中等复杂度，更复杂考虑状态管理库
