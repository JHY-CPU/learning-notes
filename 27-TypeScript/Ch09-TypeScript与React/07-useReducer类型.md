# useReducer类型

## 一、概念说明

`useReducer` 适用于复杂状态逻辑，TypeScript 可以完整地推断 action 和 state 的类型。正确类型化 reducer 函数能确保每种 action 都有对应的处理逻辑，避免运行时错误。

**核心类型**：`Reducer<State, Action>` 和 `Dispatch<Action>`。

## 二、具体用法

### 2.1 基本类型化 Reducer

```tsx
import { useReducer } from 'react';

// 定义 State 类型
interface CounterState {
  count: number;
  step: number;
}

// 定义 Action 类型（有标签联合）
type CounterAction =
  | { type: 'increment' }
  | { type: 'decrement' }
  | { type: 'reset' }
  | { type: 'setStep'; payload: number };

// 初始状态
const initialState: CounterState = { count: 0, step: 1 };

// Reducer 函数 — TypeScript 自动检查 action 分支
function reducer(state: CounterState, action: CounterAction): CounterState {
  switch (action.type) {
    case 'increment':
      return { ...state, count: state.count + state.step };
    case 'decrement':
      return { ...state, count: state.count - state.step };
    case 'reset':
      return initialState;
    case 'setStep':
      return { ...state, step: action.payload };
    // 如果缺少任何 action.type 分支，TypeScript 会报错
  }
}

function Counter() {
  const [state, dispatch] = useReducer(reducer, initialState);

  return (
    <div>
      <p>计数: {state.count}, 步长: {state.step}</p>
      <button onClick={() => dispatch({ type: 'increment' })}>+1</button>
      <button onClick={() => dispatch({ type: 'decrement' })}>-1</button>
      <button onClick={() => dispatch({ type: 'setStep', payload: 5 })}>步长5</button>
      <button onClick={() => dispatch({ type: 'reset' })}>重置</button>
    </div>
  );
}
```

### 2.2 使用 Discriminated Union 处理 Payload

```tsx
// 更复杂的 Action 定义
type TodoAction =
  | { type: 'add'; payload: { text: string } }
  | { type: 'toggle'; payload: { id: number } }
  | { type: 'delete'; payload: { id: number } }
  | { type: 'edit'; payload: { id: number; text: string } }
  | { type: 'clearCompleted' };

interface Todo {
  id: number;
  text: string;
  completed: boolean;
}

function todoReducer(state: Todo[], action: TodoAction): Todo[] {
  switch (action.type) {
    case 'add':
      return [...state, { id: Date.now(), text: action.payload.text, completed: false }];
    case 'toggle':
      return state.map(todo =>
        todo.id === action.payload.id
          ? { ...todo, completed: !todo.completed }
          : todo
      );
    case 'delete':
      return state.filter(todo => todo.id !== action.payload.id);
    case 'edit':
      return state.map(todo =>
        todo.id === action.payload.id
          ? { ...todo, text: action.payload.text }
          : todo
      );
    case 'clearCompleted':
      return state.filter(todo => !todo.completed);
  }
}
```

### 2.3 惰性初始化

```tsx
// 使用 init 函数进行惰性初始化
function init(initialCount: number): CounterState {
  return { count: initialCount, step: 1 };
}

function LazyCounter({ initialCount }: { initialCount: number }) {
  const [state, dispatch] = useReducer(reducer, initialCount, init);
  // 第三个参数 init 接受第二个参数，返回初始 state
  return <div>{state.count}</div>;
}
```

### 2.4 提取 Dispatch 类型

```tsx
// 将 dispatch 类型提取出来，用于子组件
type TodoDispatch = React.Dispatch<TodoAction>;

// 通过 Context 传递 dispatch
const TodoDispatchContext = React.createContext<TodoDispatch | null>(null);

function useTodoDispatch(): TodoDispatch {
  const dispatch = useContext(TodoDispatchContext);
  if (!dispatch) {
    throw new Error('useTodoDispatch 必须在 TodoProvider 内使用');
  }
  return dispatch;
}
```

## 三、注意事项与常见陷阱

1. **Action 必须是 Discriminated Union**：每个 action 必须有唯一的 `type` 字面量
2. **Reducer 中不要遗漏 case**：不使用 `default` 抛异常，而是让 TypeScript 检查穷尽性
3. **Payload 应内联在 action 中**：避免定义分离的 payload 类型
4. **使用 `never` 实现穷尽性检查**：
   ```typescript
   default: const _exhaustive: never = action; return state;
   ```
5. **`dispatch` 函数是稳定的引用**：可以安全地传给子组件而不需要 `useCallback`
