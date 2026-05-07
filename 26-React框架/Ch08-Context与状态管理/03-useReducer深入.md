# useReducer 深入

`useReducer` 是 React 内置的 Hook，适用于管理复杂的状态逻辑，特别是当状态更新依赖于之前的状态或涉及多个子值时。

---

## 一、基础语法

### 1.1 useReducer 基本结构

```jsx
import { useReducer } from 'react';

// reducer: 纯函数，接收当前状态和 action，返回新状态
function reducer(state, action) {
  switch (action.type) {
    case 'INCREMENT':
      return { count: state.count + 1 };
    case 'DECREMENT':
      return { count: state.count - 1 };
    case 'RESET':
      return { count: 0 };
    default:
      throw new Error(`未知 action: ${action.type}`);
  }
}

function Counter() {
  const [state, dispatch] = useReducer(reducer, { count: 0 });

  return (
    <div>
      <p>计数: {state.count}</p>
      <button onClick={() => dispatch({ type: 'INCREMENT' })}>+1</button>
      <button onClick={() => dispatch({ type: 'DECREMENT' })}>-1</button>
      <button onClick={() => dispatch({ type: 'RESET' })}>重置</button>
    </div>
  );
}
```

### 1.2 reducer 函数的规则

```jsx
// reducer 必须是纯函数：
// 1. 相同输入 → 相同输出
// 2. 不能有副作用（不能修改外部变量、不能发起请求等）
// 3. 不能修改原状态，必须返回新状态

// ❌ 错误：修改了原状态
function badReducer(state, action) {
  state.count += 1;  // 直接修改！
  return state;
}

// ❌ 错误：有副作用
function badReducer2(state, action) {
  fetch('/api/log');  // 副作用！
  return { count: state.count + 1 };
}

// ✅ 正确：返回新状态
function goodReducer(state, action) {
  return { count: state.count + 1 };
}

// ✅ 正确：使用展开运算符创建新对象
function goodReducer2(state, action) {
  return { ...state, count: state.count + 1 };
}
```

---

## 二、Action 模式

### 2.1 标准 Action 结构

```jsx
// 标准 Flux action 结构
const action = {
  type: 'ACTION_TYPE',     // 必需：动作类型
  payload: { ... },        // 可选：携带的数据
  meta: { ... },           // 可选：额外元信息
  error: true,             // 可选：是否为错误
};

// reducer 处理
function reducer(state, action) {
  switch (action.type) {
    case 'ADD_TODO':
      return {
        ...state,
        todos: [...state.todos, {
          id: action.payload.id,
          text: action.payload.text,
          done: false,
        }],
      };
    case 'TOGGLE_TODO':
      return {
        ...state,
        todos: state.todos.map(t =>
          t.id === action.payload ? { ...t, done: !t.done } : t
        ),
      };
    case 'DELETE_TODO':
      return {
        ...state,
        todos: state.todos.filter(t => t.id !== action.payload),
      };
    default:
      return state;
  }
}
```

### 2.2 Action Creator 函数

```jsx
// 封装 action 创建逻辑
const actions = {
  addTodo: (text) => ({
    type: 'ADD_TODO',
    payload: { id: Date.now(), text },
  }),

  toggleTodo: (id) => ({
    type: 'TOGGLE_TODO',
    payload: id,
  }),

  deleteTodo: (id) => ({
    type: 'DELETE_TODO',
    payload: id,
  }),

  setFilter: (filter) => ({
    type: 'SET_FILTER',
    payload: filter,
  }),
};

// 使用
dispatch(actions.addTodo('学习 useReducer'));
dispatch(actions.toggleTodo(123));
```

---

## 三、复杂状态转换

### 3.1 多字段状态管理

```jsx
const initialState = {
  // 表单字段
  name: '',
  email: '',
  password: '',
  confirmPassword: '',
  // UI 状态
  step: 1,
  errors: {},
  isSubmitting: false,
  isSuccess: false,
};

function formReducer(state, action) {
  switch (action.type) {
    case 'SET_FIELD':
      return {
        ...state,
        [action.payload.field]: action.payload.value,
        // 清除该字段的错误
        errors: { ...state.errors, [action.payload.field]: undefined },
      };

    case 'SET_ERRORS':
      return { ...state, errors: action.payload };

    case 'NEXT_STEP':
      return { ...state, step: Math.min(state.step + 1, 3) };

    case 'PREV_STEP':
      return { ...state, step: Math.max(state.step - 1, 1) };

    case 'SUBMIT':
      return { ...state, isSubmitting: true };

    case 'SUBMIT_SUCCESS':
      return { ...initialState, isSuccess: true };

    case 'SUBMIT_FAILURE':
      return { ...state, isSubmitting: false, errors: action.payload };

    case 'RESET':
      return initialState;

    default:
      return state;
  }
}

function RegistrationForm() {
  const [state, dispatch] = useReducer(formReducer, initialState);

  const handleChange = (field) => (e) => {
    dispatch({ type: 'SET_FIELD', payload: { field, value: e.target.value } });
  };

  return (
    <form>
      <input value={state.name} onChange={handleChange('name')} />
      <input value={state.email} onChange={handleChange('email')} />
      {state.errors.email && <span className="error">{state.errors.email}</span>}
      <button type="button" onClick={() => dispatch({ type: 'NEXT_STEP' })}>
        下一步
      </button>
    </form>
  );
}
```

### 3.2 嵌套状态更新

```jsx
function reducer(state, action) {
  switch (action.type) {
    case 'UPDATE_NESTED':
      return {
        ...state,
        user: {
          ...state.user,
          profile: {
            ...state.user.profile,
            [action.payload.field]: action.payload.value,
          },
        },
      };

    case 'ADD_TO_ARRAY':
      return {
        ...state,
        items: [...state.items, action.payload],
      };

    case 'UPDATE_IN_ARRAY':
      return {
        ...state,
        items: state.items.map(item =>
          item.id === action.payload.id
            ? { ...item, ...action.payload.data }
            : item
        ),
      };

    case 'REMOVE_FROM_ARRAY':
      return {
        ...state,
        items: state.items.filter(item => item.id !== action.payload),
      };

    default:
      return state;
  }
}
```

---

## 四、结合 useContext 使用

### 4.1 经典模式：Context + useReducer

这是 Redux-like 的轻量替代方案。

```jsx
// 1. 创建 Context
const TodoContext = createContext();
const TodoDispatchContext = createContext();

// 2. 定义 reducer
function todoReducer(state, action) {
  switch (action.type) {
    case 'ADD':
      return [...state, {
        id: Date.now(),
        text: action.payload,
        done: false,
      }];
    case 'TOGGLE':
      return state.map(t =>
        t.id === action.payload ? { ...t, done: !t.done } : t
      );
    case 'DELETE':
      return state.filter(t => t.id !== action.payload);
    default:
      throw new Error(`未知 action: ${action.type}`);
  }
}

// 3. Provider 组件
function TodoProvider({ children }) {
  const [todos, dispatch] = useReducer(todoReducer, []);

  return (
    <TodoContext.Provider value={todos}>
      <TodoDispatchContext.Provider value={dispatch}>
        {children}
      </TodoDispatchContext.Provider>
    </TodoContext.Provider>
  );
}

// 4. 自定义 Hooks
function useTodos() {
  const context = useContext(TodoContext);
  if (context === undefined) {
    throw new Error('useTodos 必须在 TodoProvider 内使用');
  }
  return context;
}

function useTodoDispatch() {
  const context = useContext(TodoDispatchContext);
  if (context === undefined) {
    throw new Error('useTodoDispatch 必须在 TodoProvider 内使用');
  }
  return context;
}

// 5. 使用（组件不需要知道状态管理方式）
function TodoList() {
  const todos = useTodos();
  return (
    <ul>
      {todos.map(todo => <TodoItem key={todo.id} todo={todo} />)}
    </ul>
  );
}

function AddTodo() {
  const dispatch = useTodoDispatch();
  const [text, setText] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    dispatch({ type: 'ADD', payload: text });
    setText('');
  };

  return (
    <form onSubmit={handleSubmit}>
      <input value={text} onChange={e => setText(e.target.value)} />
      <button type="submit">添加</button>
    </form>
  );
}

// dispatch 是稳定的引用，不因状态变化而变化
// 因此 AddTodo 不会在 todos 变化时重渲染
```

### 4.2 为什么拆分 State 和 Dispatch

```jsx
// 不拆分：state 变化时所有消费者都重渲染
const TodoContext = createContext();

function TodoProvider({ children }) {
  const [todos, dispatch] = useReducer(reducer, []);
  return (
    <TodoContext.Provider value={{ todos, dispatch }}>
      {children}
    </TodoContext.Provider>
  );
}

// AddTodo 也会因 todos 变化而重渲染（虽然它不使用 todos）
function AddTodo() {
  const { dispatch } = useContext(TodoContext);  // 每次 value 对象变了就重渲染
  // ...
}

// 拆分后：只使用 dispatch 的组件不受 state 变化影响
<TodoContext.Provider value={todos}>
  <TodoDispatchContext.Provider value={dispatch}>
    {children}
  </TodoDispatchContext.Provider>
</TodoContext.Provider>
```

---

## 五、useReducer vs useState

### 对比

| 特性 | useState | useReducer |
|---|---|---|
| 状态复杂度 | 简单值 | 复杂对象/多个子值 |
| 更新逻辑 | 分散在各处 | 集中在 reducer |
| 可测试性 | 较差 | reducer 是纯函数，易于测试 |
| 可读性 | 简单场景更好 | 复杂逻辑更清晰 |
| 调试 | 一般 | 可以记录所有 action |
| 学习曲线 | 低 | 中等 |

### 选择指南

```
状态是单个简单值？───────► useState
状态更新逻辑简单？───────► useState
状态有多个子值？─────────► useReducer
下一个状态依赖上一个？───► useReducer（两者都行，但 reducer 更清晰）
更新逻辑复杂？───────────► useReducer
需要共享更新逻辑？───────► useReducer + Context
```

### 示例对比

```jsx
// useState: 简单场景
function Counter() {
  const [count, setCount] = useState(0);
  return <button onClick={() => setCount(c => c + 1)}>计数: {count}</button>;
}

// useReducer: 复杂场景
function ShoppingCart() {
  const [state, dispatch] = useReducer(cartReducer, { items: [], total: 0 });

  return (
    <div>
      {state.items.map(item => (
        <div key={item.id}>
          {item.name} x{item.quantity}
          <button onClick={() => dispatch({ type: 'REMOVE', payload: item.id })}>
            移除
          </button>
        </div>
      ))}
      <p>总计: ¥{state.total}</p>
    </div>
  );
}
```

---

## 六、Reducer 组合

当 reducer 过于庞大时，拆分为多个子 reducer。

```jsx
// 子 reducer
function userReducer(state, action) {
  switch (action.type) {
    case 'SET_USER': return { ...state, profile: action.payload };
    case 'LOGOUT':   return { ...state, profile: null, preferences: {} };
    case 'UPDATE_PREF': return {
      ...state,
      preferences: { ...state.preferences, ...action.payload },
    };
    default: return state;
  }
}

function cartReducer(state, action) {
  switch (action.type) {
    case 'ADD_ITEM':    return [...state, action.payload];
    case 'REMOVE_ITEM': return state.filter(i => i.id !== action.payload);
    case 'CLEAR':       return [];
    default: return state;
  }
}

// 组合 reducer
function appReducer(state, action) {
  return {
    user: userReducer(state.user, action),
    cart: cartReducer(state.cart, action),
  };
}

// 使用
const initialState = {
  user: { profile: null, preferences: {} },
  cart: [],
};

function App() {
  const [state, dispatch] = useReducer(appReducer, initialState);
  // dispatch 可以同时作用于 user 和 cart
  // 两个 reducer 都会收到每个 action，不需要的忽略即可
}
```

---

## 七、TypeScript 集成

```tsx
// 定义 State 类型
interface State {
  count: number;
  user: User | null;
  loading: boolean;
  error: string | null;
}

// 定义所有可能的 Action 类型（联合类型）
type Action =
  | { type: 'INCREMENT' }
  | { type: 'DECREMENT' }
  | { type: 'SET_COUNT'; payload: number }
  | { type: 'SET_USER'; payload: User }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'RESET' };

// reducer 函数
function reducer(state: State, action: Action): State {
  switch (action.type) {
    case 'INCREMENT':
      return { ...state, count: state.count + 1 };
    case 'DECREMENT':
      return { ...state, count: state.count - 1 };
    case 'SET_COUNT':
      return { ...state, count: action.payload };
    case 'SET_USER':
      return { ...state, user: action.payload, error: null };
    case 'SET_LOADING':
      return { ...state, loading: action.payload };
    case 'SET_ERROR':
      return { ...state, error: action.payload, loading: false };
    case 'RESET':
      return initialState;
    default:
      // TypeScript 的穷尽性检查
      const _exhaustive: never = action;
      return state;
  }
}

// Action Creator 辅助函数（带类型推断）
const actions = {
  increment: (): Action => ({ type: 'INCREMENT' }),
  decrement: (): Action => ({ type: 'DECREMENT' }),
  setCount: (count: number): Action => ({ type: 'SET_COUNT', payload: count }),
  setUser: (user: User): Action => ({ type: 'SET_USER', payload: user }),
  setLoading: (loading: boolean): Action => ({ type: 'SET_LOADING', payload: loading }),
  setError: (error: string | null): Action => ({ type: 'SET_ERROR', payload: error }),
  reset: (): Action => ({ type: 'RESET' }),
};

// 使用
function Counter() {
  const [state, dispatch] = useReducer(reducer, initialState);

  return (
    <div>
      <span>{state.count}</span>
      <button onClick={() => dispatch(actions.increment())}>+1</button>
      <button onClick={() => dispatch(actions.setCount(0))}>重置</button>
    </div>
  );
}
```

### 使用 useReducer 的 init 函数

```tsx
// init 函数用于惰性初始化：只在首次渲染时执行
interface State {
  count: number;
}

function init(initialCount: number): State {
  // 可以包含复杂计算逻辑
  // 只在组件挂载时执行一次
  return { count: initialCount };
}

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case 'INCREMENT': return { count: state.count + 1 };
    case 'DECREMENT': return { count: state.count - 1 };
    case 'RESET':     return init(action.payload);
    default:          return state;
  }
}

function Counter({ initialCount }: { initialCount: number }) {
  // 第三个参数 init 是可选的
  // init(initialCount) 的返回值作为初始状态
  const [state, dispatch] = useReducer(reducer, initialCount, init);

  return (
    <div>
      <p>计数: {state.count}</p>
      <button onClick={() => dispatch({ type: 'RESET', payload: initialCount })}>
        重置
      </button>
    </div>
  );
}
```

---

## 八、调试与测试

### 8.1 添加日志中间件

```jsx
function withLogging(reducer) {
  return (state, action) => {
    console.group(`Action: ${action.type}`);
    console.log('Previous State:', state);
    console.log('Action:', action);

    const nextState = reducer(state, action);

    console.log('Next State:', nextState);
    console.groupEnd();
    return nextState;
  };
}

// 使用
const [state, dispatch] = useReducer(
  withLogging(reducer),
  initialState
);
```

### 8.2 测试 reducer

```jsx
// reducer 是纯函数，非常容易测试
describe('todoReducer', () => {
  it('应该添加新任务', () => {
    const state = [];
    const action = { type: 'ADD', payload: { id: 1, text: '学习' } };
    const result = todoReducer(state, action);

    expect(result).toHaveLength(1);
    expect(result[0].text).toBe('学习');
  });

  it('应该切换任务状态', () => {
    const state = [{ id: 1, text: '学习', done: false }];
    const action = { type: 'TOGGLE', payload: 1 };
    const result = todoReducer(state, action);

    expect(result[0].done).toBe(true);
  });

  it('不应该修改原状态', () => {
    const state = [{ id: 1, text: '学习', done: false }];
    const action = { type: 'TOGGLE', payload: 1 };
    const result = todoReducer(state, action);

    expect(state[0].done).toBe(false);  // 原状态不变
    expect(result).not.toBe(state);      // 返回新对象
  });
});
```
