# Context API 详解

Context 提供了一种在组件树中共享数据的方式，无需逐层传递 props（避免 "prop drilling"）。

---

## 一、createContext

### 1.1 创建 Context

```jsx
import { createContext } from 'react';

// createContext 接受一个默认值参数
const ThemeContext = createContext('light');

// 最佳实践：提供有意义的默认值
const AuthContext = createContext({
  user: null,
  login: () => {},
  logout: () => {},
});

// 导出供其他组件使用
export { ThemeContext, AuthContext };
```

### 1.2 默认值的生效时机

**默认值只在没有匹配的 Provider 时生效**：

```jsx
const ColorContext = createContext('red');

// 组件树中没有 <ColorContext.Provider> 包裹时
function DeepChild() {
  const color = useContext(ColorContext);
  return <span style={{ color }}>{color}</span>; // 输出 "red"
}

// 有 Provider 但未提供 value 时，不会使用默认值！
function App() {
  return (
    <ColorContext.Provider>
      <DeepChild />
      {/* value 为 undefined，不是 "red"！ */}
    </ColorContext.Provider>
  );
}
```

### 1.3 Context 的 TypeScript 类型

```tsx
import { createContext, useContext } from 'react';

interface User {
  id: string;
  name: string;
  role: 'admin' | 'user' | 'guest';
}

interface AuthContextType {
  user: User | null;
  login: (credentials: { email: string; password: string }) => Promise<void>;
  logout: () => void;
  loading: boolean;
}

// 方式 1: 提供完整默认值（推荐）
const AuthContext = createContext<AuthContextType>({
  user: null,
  login: async () => {},
  logout: () => {},
  loading: false,
});

// 方式 2: null 作为默认值，需要类型断言和守卫
const AuthContext2 = createContext<AuthContextType | null>(null);

function useAuth() {
  const context = useContext(AuthContext2);
  if (!context) {
    throw new Error('useAuth 必须在 AuthProvider 内使用');
  }
  return context;
}
```

---

## 二、Provider

### 2.1 基础用法

```jsx
function App() {
  const theme = 'dark';

  return (
    <ThemeContext.Provider value={theme}>
      <Toolbar />
    </ThemeContext.Provider>
  );
}
```

### 2.2 Provider value 可以是任何值

```jsx
// 对象
const ConfigContext = createContext();

function App() {
  const config = {
    apiUrl: 'https://api.example.com',
    version: '2.0',
    features: { darkMode: true, notifications: true },
  };

  return (
    <ConfigContext.Provider value={config}>
      <MainApp />
    </ConfigContext.Provider>
  );
}

// 数组
const [state, dispatch] = useReducer(reducer, initialState);
return (
  <StateContext.Provider value={[state, dispatch]}>
    <App />
  </StateContext.Provider>
);
```

### 2.3 带状态和操作的 Provider

```jsx
const AuthContext = createContext();

function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // 检查本地存储中的 token
    const token = localStorage.getItem('token');
    if (token) {
      fetchUser(token).then(setUser).finally(() => setLoading(false));
    } else {
      setLoading(false);
    }
  }, []);

  const login = async (email, password) => {
    const { token, user } = await apiLogin(email, password);
    localStorage.setItem('token', token);
    setUser(user);
  };

  const logout = () => {
    localStorage.removeItem('token');
    setUser(null);
  };

  // value 包含状态和操作函数
  const value = { user, loading, login, logout };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}
```

---

## 三、useContext

### 3.1 基础用法

```jsx
import { useContext } from 'react';
import { ThemeContext } from './contexts';

function ThemedButton() {
  const theme = useContext(ThemeContext);

  return (
    <button className={`btn btn-${theme}`}>
      当前主题: {theme}
    </button>
  );
}
```

### 3.2 消费多个 Context

```jsx
function Dashboard() {
  const theme = useContext(ThemeContext);
  const { user, logout } = useContext(AuthContext);
  const config = useContext(ConfigContext);

  return (
    <div className={`dashboard-${theme}`}>
      <h1>欢迎, {user.name}</h1>
      <p>版本: {config.version}</p>
      <button onClick={logout}>退出</button>
    </div>
  );
}
```

### 3.3 自定义 Hook 封装 Context

```jsx
// 封装后提供更好的错误处理和类型推断
function useTheme() {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme 必须在 ThemeProvider 内使用');
  }
  return context;
}

function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth 必须在 AuthProvider 内使用');
  }
  return context;
}

// 使用
function Header() {
  const theme = useTheme();
  const { user, logout } = useAuth();

  return <header className={theme}>...</header>;
}
```

---

## 四、嵌套 Provider

### 4.1 多 Context 嵌套

```jsx
function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <ConfigProvider>
          <I18nProvider>
            <AppContent />
          </I18nProvider>
        </ConfigProvider>
      </AuthProvider>
    </ThemeProvider>
  );
}
```

### 4.2 组合 Provider 避免嵌套地狱

```jsx
// 创建一个组合所有 Provider 的组件
function AppProviders({ children }) {
  return (
    <ThemeProvider>
      <AuthProvider>
        <ConfigProvider>
          <I18nProvider>
            {children}
          </I18nProvider>
        </ConfigProvider>
      </AuthProvider>
    </ThemeProvider>
  );
}

// 使用
function App() {
  return (
    <AppProviders>
      <AppContent />
    </AppProviders>
  );
}
```

### 4.3 动态组合 Provider

```jsx
// 更灵活的组合方式
function composeProviders(...providers) {
  return providers.reduce(
    (Prev, Curr) => ({ children }) => (
      <Prev>
        <Curr>{children}</Curr>
      </Prev>
    )
  );
}

const AppProviders = composeProviders(
  ThemeProvider,
  AuthProvider,
  ConfigProvider,
  I18nProvider,
);

function App() {
  return (
    <AppProviders>
      <AppContent />
    </AppProviders>
  );
}
```

---

## 五、Context 组合模式

### 5.1 按领域拆分 Context

```jsx
// 用户相关
const UserContext = createContext();

function UserProvider({ children }) {
  const [user, setUser] = useState(null);
  const [preferences, setPreferences] = useState({});

  const value = { user, setUser, preferences, setPreferences };
  return <UserContext.Provider value={value}>{children}</UserContext.Provider>;
}

// UI 相关
const UIContext = createContext();

function UIProvider({ children }) {
  const [theme, setTheme] = useState('light');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [notifications, setNotifications] = useState([]);

  const value = { theme, setTheme, sidebarOpen, setSidebarOpen, notifications, setNotifications };
  return <UIContext.Provider value={value}>{children}</UIContext.Provider>;
}
```

### 5.2 Context 中存储派生状态

```jsx
const TodoContext = createContext();

function TodoProvider({ children }) {
  const [todos, setTodos] = useState([]);

  // 派生状态直接放在 value 中
  const value = {
    todos,
    addTodo: (text) => setTodos(prev => [...prev, { id: Date.now(), text, done: false }]),
    toggleTodo: (id) => setTodos(prev =>
      prev.map(t => t.id === id ? { ...t, done: !t.done } : t)
    ),
    removeTodo: (id) => setTodos(prev => prev.filter(t => t.id !== id)),
    // 派生数据
    completedCount: todos.filter(t => t.done).length,
    totalCount: todos.length,
    allCompleted: todos.length > 0 && todos.every(t => t.done),
  };

  return <TodoContext.Provider value={value}>{children}</TodoContext.Provider>;
}
```

---

## 六、性能问题

### 6.1 每个 Consumer 都会重新渲染

**核心问题**：当 Provider 的 `value` 变化时，所有使用 `useContext` 的组件都会重新渲染，即使它们只使用了 value 中未变化的部分。

```jsx
const AppContext = createContext();

function App() {
  const [user, setUser] = useState(null);
  const [theme, setTheme] = useState('light');

  const value = { user, setUser, theme, setTheme };

  return (
    <AppContext.Provider value={value}>
      <UserProfile />  {/* 只用 user，但 theme 变了也会重渲染 */}
      <ThemeToggle />  {/* 只用 theme，但 user 变了也会重渲染 */}
    </AppContext.Provider>
  );
}
```

> **详见**：[02-Context性能优化.md](./02-Context性能优化.md)

---

## 七、Context vs Prop Drilling 决策树

```
需要共享数据吗？
├── 否 → 普通 props 传递
└── 是 → 数据的使用范围？
    ├── 仅父子组件 → Props
    ├── 少数几层 → Props（prop drilling 在3层内可接受）
    └── 多层/深层 → Context
        ├── 数据变化频率？
        │   ├── 低频（主题、配置、认证）→ Context ✅
        │   ├── 中频（表单状态）→ Context + useMemo ⚠️
        │   └── 高频（动画、鼠标位置）→ 避免 Context ❌
        └── 数据类型？
            ├── 只读数据 → Context ✅
            ├── 读写数据 → Context + useReducer
            └── 复杂全局状态 → 状态管理库
```

### 适用场景

| 场景 | 推荐方案 |
|---|---|
| 主题切换 | Context ✅ |
| 用户认证 | Context ✅ |
| 国际化 (i18n) | Context ✅ |
| 表单状态 | Context + useReducer 或 form 库 |
| 购物车 | Context + useReducer |
| 实时数据流 | 状态管理库 / WebSocket |
| 鼠标位置、滚动位置 | 避免 Context（高频更新） |

### Prop Drilling 的问题

```jsx
// Prop drilling: 需要经过中间层传递
function App() {
  const theme = 'dark';
  return <Layout theme={theme} />;
}

function Layout({ theme }) {          // 不需要 theme，但必须传递
  return <Content theme={theme} />;
}

function Content({ theme }) {         // 不需要 theme，但必须传递
  return <Button theme={theme} />;    // 终于用到了
}

// Context: 直接在需要的地方获取
function Button() {
  const theme = useContext(ThemeContext);  // 直接获取，中间组件不需要关心
  return <button className={theme}>按钮</button>;
}
```

---

## 八、完整示例：主题系统

```jsx
import { createContext, useContext, useState, useCallback, useMemo } from 'react';

// 1. 定义主题配置
const themes = {
  light: {
    name: 'light',
    colors: { bg: '#fff', text: '#333', primary: '#007bff', border: '#dee2e6' },
    font: { family: 'system-ui', size: '16px' },
  },
  dark: {
    name: 'dark',
    colors: { bg: '#1a1a2e', text: '#eee', primary: '#6c63ff', border: '#333' },
    font: { family: 'system-ui', size: '16px' },
  },
};

// 2. 创建 Context
const ThemeContext = createContext(undefined);

// 3. 创建 Provider
function ThemeProvider({ children, defaultTheme = 'light' }) {
  const [themeName, setThemeName] = useState(defaultTheme);

  const toggleTheme = useCallback(() => {
    setThemeName(prev => prev === 'light' ? 'dark' : 'light');
  }, []);

  const setTheme = useCallback((name) => {
    if (themes[name]) setThemeName(name);
  }, []);

  // useMemo 确保 theme 对象不变（除非 themeName 变）
  const value = useMemo(() => ({
    theme: themes[themeName],
    themeName,
    toggleTheme,
    setTheme,
  }), [themeName, toggleTheme, setTheme]);

  return (
    <ThemeContext.Provider value={value}>
      <div data-theme={themeName}>
        {children}
      </div>
    </ThemeContext.Provider>
  );
}

// 4. 自定义 Hook
function useTheme() {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme 必须在 ThemeProvider 内使用');
  }
  return context;
}

// 5. 使用
function App() {
  return (
    <ThemeProvider defaultTheme="light">
      <Header />
      <MainContent />
    </ThemeProvider>
  );
}

function Header() {
  const { theme, toggleTheme } = useTheme();
  return (
    <header style={{ background: theme.colors.bg, color: theme.colors.text }}>
      <h1>我的应用</h1>
      <button onClick={toggleTheme}>
        切换到{theme.name === 'light' ? '深色' : '浅色'}模式
      </button>
    </header>
  );
}

function MainContent() {
  const { theme } = useTheme();
  return (
    <main style={{ background: theme.colors.bg, color: theme.colors.text }}>
      <p>当前主题: {theme.name}</p>
    </main>
  );
}
```

---

## 九、常见陷阱

### 陷阱 1: 在 Provider 内部创建对象导致不必要的重渲染

```jsx
// 错误：每次 App 渲染都创建新的 value 对象
function App() {
  return (
    <ThemeContext.Provider value={{ theme: 'dark', toggle: () => {} }}>
      <Content />
    </ThemeContext.Provider>
  );
}

// 正确：用 useMemo 稳定 value
function App() {
  const [theme, setTheme] = useState('dark');
  const toggle = useCallback(() => setTheme(t => t === 'dark' ? 'light' : 'dark'), []);

  const value = useMemo(() => ({ theme, toggle }), [theme, toggle]);

  return (
    <ThemeContext.Provider value={value}>
      <Content />
    </ThemeContext.Provider>
  );
}
```

### 陷阱 2: 忘记包裹 Provider

```jsx
// 没有 Provider 时使用 useContext 返回默认值
const AuthContext = createContext(null);

function UserProfile() {
  const auth = useContext(AuthContext);
  // auth 为 null（默认值），不是 Provider 中的值
  return <div>{auth?.user?.name}</div>;  // 需要做安全检查
}
```

### 陷阱 3: 一个 Context 存放太多数据

```jsx
// 不好：一个巨大的 Context
const AppContext = createContext({
  user: null,
  theme: 'light',
  locale: 'zh',
  notifications: [],
  cart: [],
  settings: {},
  // ...50 more properties
});

// 好：按领域拆分
const UserContext = createContext();
const ThemeContext = createContext();
const CartContext = createContext();
```
