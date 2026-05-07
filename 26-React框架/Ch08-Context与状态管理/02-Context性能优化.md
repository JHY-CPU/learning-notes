# Context 性能优化

Context 的最大问题是：Provider 的 value 变化时，所有消费者组件都会重新渲染，即使消费者只使用了 value 中未变化的部分。本文详细讲解各种优化策略。

---

## 一、问题分析

```jsx
const AppContext = createContext();

function App() {
  const [user, setUser] = useState(null);
  const [theme, setTheme] = useState('light');
  const [count, setCount] = useState(0);

  // 每次任何状态变化，value 都是新对象
  const value = { user, theme, count, setUser, setTheme, setCount };

  return (
    <AppContext.Provider value={value}>
      <UserProfile />  {/* 只用 user */}
      <ThemeToggle />  {/* 只用 theme */}
      <Counter />      {/* 只用 count */}
    </AppContext.Provider>
  );
}

// 每个消费者组件：任何 value 变化都会重渲染
function UserProfile() {
  const { user } = useContext(AppContext);
  console.log('UserProfile 渲染');  // count 变化时也会打印！
  return <div>{user?.name}</div>;
}
```

---

## 二、useMemo 稳定 Provider value

最基本的优化：确保 value 对象引用稳定。

```jsx
function App() {
  const [user, setUser] = useState(null);
  const [theme, setTheme] = useState('light');

  const login = useCallback((email, password) => { /* ... */ }, []);
  const logout = useCallback(() => { /* ... */ }, []);
  const toggleTheme = useCallback(() => {
    setTheme(t => t === 'light' ? 'dark' : 'light');
  }, []);

  // useMemo 确保只有依赖变化时才创建新对象
  const value = useMemo(() => ({
    user,
    theme,
    login,
    logout,
    toggleTheme,
  }), [user, theme, login, logout, toggleTheme]);

  return (
    <AppContext.Provider value={value}>
      {children}
    </AppContext.Provider>
  );
}
```

> **注意**：useMemo 只能保证 value 引用不变，不能阻止消费者的重渲染——只要 value 的任何属性变了，value 引用就会变。

---

## 三、拆分 Context

最有效的优化策略：将不同关注点的数据放在不同的 Context 中。

```jsx
// 拆分前：一个 Context 包含所有数据
const AppContext = createContext();  // user 变化 → 所有消费者重渲染

// 拆分后：每个关注点一个 Context
const UserContext = createContext();   // user 变化 → 只有 user 消费者重渲染
const ThemeContext = createContext();  // theme 变化 → 只有 theme 消费者重渲染
const CartContext = createContext();   // cart 变化 → 只有 cart 消费者重渲染
```

### 完整示例

```jsx
// contexts/index.jsx
export const UserContext = createContext(null);
export const ThemeContext = createContext(null);
export const CartContext = createContext(null);

// providers/UserProvider.jsx
export function UserProvider({ children }) {
  const [user, setUser] = useState(null);

  const value = useMemo(() => ({ user, setUser }), [user]);

  return <UserContext.Provider value={value}>{children}</UserContext.Provider>;
}

// providers/ThemeProvider.jsx
export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState('light');

  const value = useMemo(() => ({ theme, setTheme }), [theme]);

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

// providers/CartProvider.jsx
export function CartProvider({ children }) {
  const [items, setItems] = useState([]);

  const value = useMemo(() => ({
    items,
    addItem: (item) => setItems(prev => [...prev, item]),
    removeItem: (id) => setItems(prev => prev.filter(i => i.id !== id)),
    total: items.reduce((sum, i) => sum + i.price * i.quantity, 0),
  }), [items]);

  return <CartContext.Provider value={value}>{children}</CartContext.Provider>;
}
```

---

## 四、Context Selector 模式

Context Selector 允许消费者只订阅 value 的一部分，而不是整个 value。

### 4.1 使用 use-context-selector 库

```bash
npm install use-context-selector
```

```jsx
import { createContext } from 'use-context-selector';
import { useContextSelector } from 'use-context-selector';

const AppContext = createContext();

function AppProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const value = useMemo(() => ({ state, dispatch }), [state]);

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

// 只订阅需要的部分 —— selector 返回值不变时不会重渲染
function UserProfile() {
  const user = useContextSelector(AppContext, v => v.state.user);
  return <div>{user.name}</div>;
}

function ThemeToggle() {
  const theme = useContextSelector(AppContext, v => v.state.theme);
  const dispatch = useContextSelector(AppContext, v => v.dispatch);
  return <button onClick={() => dispatch({ type: 'TOGGLE_THEME' })}>{theme}</button>;
}
```

### 4.2 手动实现 useMemoizedSelector

```jsx
import { useContext, useRef, useSyncExternalStore } from 'react';

function useMemoizedSelector(context, selector) {
  const selectedRef = useRef();

  return useSyncExternalStore(
    // subscribe: 当 context 变化时通知
    (callback) => {
      // Context 不提供订阅机制，这里简化处理
      return () => {};
    },
    // getSnapshot: 返回选中的值
    () => {
      const contextValue = useContext(context);
      const selected = selector(contextValue);
      if (selectedRef.current !== selected) {
        selectedRef.current = selected;
      }
      return selectedRef.current;
    }
  );
}
```

### 4.3 手动封装 Context + Selector

```jsx
// 通过拆分 value 到独立的 Context 实现类似效果
function createSplitContext() {
  const StateContext = createContext();
  const DispatchContext = createContext();

  function Provider({ children, reducer, initialState }) {
    const [state, dispatch] = useReducer(reducer, initialState);

    // 状态和 dispatch 分开提供
    // dispatch 引用永远不变，不会引起额外渲染
    return (
      <DispatchContext.Provider value={dispatch}>
        <StateContext.Provider value={state}>
          {children}
        </StateContext.Provider>
      </DispatchContext.Provider>
    );
  }

  // 自定义 hooks
  const useState2 = () => useContext(StateContext);
  const useDispatch = () => useContext(DispatchContext);

  // 用 useMemo 实现 selector
  const useSelector = (selector) => {
    const state = useContext(StateContext);
    return useMemo(() => selector(state), [state]);
  };

  return { Provider, useState: useState2, useDispatch, useSelector };
}

// 使用
const { Provider: AppProvider, useSelector, useDispatch } = createSplitContext();

function Counter() {
  // 只在 count 变化时重渲染，不关心 state 中其他字段
  const count = useSelector(state => state.count);
  const dispatch = useDispatch();

  return (
    <div>
      <span>{count}</span>
      <button onClick={() => dispatch({ type: 'INCREMENT' })}>+1</button>
    </div>
  );
}
```

---

## 五、React.memo 优化消费者

用 `React.memo` 包裹消费者组件，配合稳定的 props 减少重渲染。

```jsx
// 不会因父组件重渲染而重渲染（props 稳定时）
const UserAvatar = React.memo(function UserAvatar({ userId }) {
  const { users } = useContext(UserContext);
  const user = users.find(u => u.id === userId);

  return <img src={user.avatar} alt={user.name} />;
});

// 父组件变化时，如果 userId 没变，UserAvatar 不会重渲染
function UserList({ userIds }) {
  return (
    <div>
      {userIds.map(id => <UserAvatar key={id} userId={id} />)}
    </div>
  );
}
```

---

## 六、将 dispatch 和 state 分离

`dispatch` 函数的引用永远不变，可以单独提供，避免因状态变化导致只使用 dispatch 的组件重渲染。

```jsx
const StateContext = createContext();
const DispatchContext = createContext();

function AppProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState);

  return (
    <DispatchContext.Provider value={dispatch}>
      <StateContext.Provider value={state}>
        {children}
      </StateContext.Provider>
    </DispatchContext.Provider>
  );
}

// 只需要 dispatch 的组件 —— 不会因 state 变化而重渲染
function AddTodoButton() {
  const dispatch = useContext(DispatchContext);
  console.log('AddTodoButton 渲染');  // state 变化不会触发

  return (
    <button onClick={() => dispatch({ type: 'ADD_TODO', text: '新任务' })}>
      添加
    </button>
  );
}

// 需要 state 的组件 —— state 变化时正常重渲染
function TodoList() {
  const state = useContext(StateContext);
  return state.todos.map(t => <div key={t.id}>{t.text}</div>);
}
```

---

## 七、使用 Zustand/Valtio 替代

当 Context 性能优化成本过高时，考虑使用外部状态管理库。

### Zustand 作为 Context 替代

```jsx
import { create } from 'zustand';

// 创建 store（不需要 Provider）
const useAppStore = create((set) => ({
  user: null,
  theme: 'light',
  setUser: (user) => set({ user }),
  toggleTheme: () => set((s) => ({ theme: s.theme === 'light' ? 'dark' : 'light' })),
}));

// 组件只在选择的数据变化时重渲染
function UserProfile() {
  // 只订阅 user —— theme 变化不触发重渲染
  const user = useAppStore((s) => s.user);
  const setUser = useAppStore((s) => s.setUser);

  return <div>{user?.name} <button onClick={() => setUser(null)}>退出</button></div>;
}

function ThemeToggle() {
  // 只订阅 theme —— user 变化不触发重渲染
  const theme = useAppStore((s) => s.theme);
  const toggleTheme = useAppStore((s) => s.toggleTheme);

  return <button onClick={toggleTheme}>{theme}</button>;
}
```

### Valtio 代理状态

```jsx
import { proxy, useSnapshot } from 'valtio';

const store = proxy({
  user: null,
  theme: 'light',
  cart: [],
});

function UserProfile() {
  // useSnapshot 自动追踪访问的属性
  const snap = useSnapshot(store);
  return <div>{snap.user?.name}</div>;
  // 只有 snap.user 变化时才重渲染
}
```

---

## 八、总结与对比

| 策略 | 难度 | 效果 | 适用场景 |
|---|---|---|---|
| useMemo 稳定 value | 低 | 中 | 所有 Context |
| 拆分 Context | 中 | 高 | 数据关注点清晰 |
| State/Dispatch 分离 | 低 | 中 | useReducer 模式 |
| Context Selector | 中 | 高 | 复杂 state 对象 |
| React.memo 消费者 | 低 | 中 | 纯展示组件 |
| Zustand 替代 | 中 | 高 | 全局状态复杂 |
| Valtio 替代 | 低 | 高 | 需要细粒度响应式 |

### 推荐实践

1. **优先拆分 Context**：按关注点拆分是最简单有效的优化
2. **dispatch 单独提供**：使用 useReducer 时天然拥有稳定的 dispatch
3. **useMemo 包裹 value**：基本功，减少不必要的对象创建
4. **超过 3 个 Context 需要考虑外部库**：复杂状态管理用 Zustand 更简洁
