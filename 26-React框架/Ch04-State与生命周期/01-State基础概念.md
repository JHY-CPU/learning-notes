# State 基础概念

## 1. 什么是 State

State 是组件内部管理的、可以随时间变化的数据。与 Props 不同，State 由组件自身创建和维护，组件是 state 的**唯一所有者**。

```
Props：从父组件接收，只读
State：组件自己创建，可变（通过 setState）
```

State 的变化会触发组件的**重新渲染**（re-render），React 会根据新的 state 计算出新的 UI 并更新 DOM。

---

## 2. useState Hook 基础

### 2.1 语法

```jsx
const [state, setState] = React.useState(initialValue);
```

- `state`：当前状态值
- `setState`：更新状态的函数
- `initialValue`：状态的初始值（仅首次渲染时使用）

### 2.2 基本示例

```jsx
import React from 'react';

function Counter() {
  const [count, setCount] = React.useState(0);

  return (
    <div>
      <p>You clicked {count} times</p>
      <button onClick={() => setCount(count + 1)}>
        Click me
      </button>
    </div>
  );
}
```

### 2.3 更新状态的两种方式

```jsx
function Counter() {
  const [count, setCount] = React.useState(0);

  // 方式一：直接传入新值
  const increment = () => setCount(count + 1);

  // 方式二：传入函数（推荐，尤其在依赖旧值时）
  const incrementWithFn = () => setCount(prev => prev + 1);

  return <button onClick={increment}>Count: {count}</button>;
}
```

---

## 3. State 的不可变性

**永远不要直接修改 state**，必须创建新的值来替换。

```jsx
// 错误：直接修改
const [user, setUser] = React.useState({ name: 'Alice', age: 25 });

user.name = 'Bob';        // 错！React 不知道 state 变了
setUser(user);            // 即使调用 setUser，因为引用相同，不会触发重渲染

// 正确：创建新对象
setUser({ ...user, name: 'Bob' });  // 新引用，触发重渲染
```

```jsx
// 错误：直接修改数组
const [items, setItems] = React.useState([1, 2, 3]);

items.push(4);            // 错！直接修改了原数组
setItems(items);

// 正确：创建新数组
setItems([...items, 4]);  // 展开旧数组 + 新元素
```

---

## 4. 函数式更新（Functional Updates）

当新 state 依赖于旧 state 时，使用函数式更新可以避免闭包陷阱。

### 4.1 问题：闭包陷阱

```jsx
function Counter() {
  const [count, setCount] = React.useState(0);

  const handleClick = () => {
    // 这里的 count 是渲染时捕获的值
    setCount(count + 1);  // 如果快速点击，可能都是基于同一个 count 值
    setCount(count + 1);  // 连续调用两次，结果只加 1，不是加 2
  };

  return <button onClick={handleClick}>Count: {count}</button>;
}
```

### 4.2 解决：函数式更新

```jsx
const handleClick = () => {
  setCount(prev => prev + 1);  // prev 总是最新的 state
  setCount(prev => prev + 1);  // 连续调用两次，正确地加 2
};
```

### 4.3 何时使用函数式更新

- 连续多次更新同一个 state
- 在 `setTimeout`、`setInterval` 等异步回调中更新
- 在事件处理函数中需要基于前一个值计算新值
- 作为依赖数组中某个值的替代方案

```jsx
// 异步场景
React.useEffect(() => {
  const timer = setInterval(() => {
    setCount(prev => prev + 1);  // 必须用函数式，否则 count 永远是 0
  }, 1000);
  return () => clearInterval(timer);
}, []);  // 空依赖数组
```

---

## 5. 延迟初始化（Lazy Initialization）

如果初始 state 的计算代价很高，可以传入一个**函数**而不是值，React 只会在首次渲染时调用它。

```jsx
// 不好：每次渲染都执行读取 localStorage
const [value, setValue] = React.useState(
  localStorage.getItem('key') || 'default'
);

// 好：只在首次渲染时执行
const [value, setValue] = React.useState(() => {
  return localStorage.getItem('key') || 'default';
});

// 另一个例子：复杂计算
const [matrix, setMatrix] = React.useState(() => {
  return createExpensiveMatrix(1000, 1000);  // 只执行一次
});
```

---

## 6. React 18 中的状态批处理（State Batching）

### 6.1 什么是批处理

React 18 中，**所有事件处理函数中的多次 state 更新会被合并为一次重新渲染**，即使跨越了异步边界。

```jsx
function Counter() {
  const [count, setCount] = React.useState(0);
  const [flag, setFlag] = React.useState(false);

  const handleClick = () => {
    setCount(c => c + 1);  // 不会立即重渲染
    setFlag(f => !f);      // 不会立即重渲染
    // React 合并为一次重渲染
  };

  console.log('rendered');  // 点击后只打印一次

  return (
    <button onClick={handleClick}>
      Count: {count}, Flag: {String(flag)}
    </button>
  );
}
```

### 6.2 React 18 之前的批处理

React 17 及以前，只在**React 事件处理函数**中批处理。在 `setTimeout`、`Promise`、原生事件中不批处理：

```jsx
// React 17：两次渲染
setTimeout(() => {
  setCount(c => c + 1);  // 渲染
  setFlag(f => !f);      // 渲染
}, 1000);

// React 18：一次渲染（自动批处理）
setTimeout(() => {
  setCount(c => c + 1);  // 合并
  setFlag(f => !f);      // 合并
}, 1000);
```

### 6.3 跳过批处理：flushSync

如果需要强制立即更新，使用 `flushSync`：

```jsx
import { flushSync } from 'react-dom';

function handleClick() {
  flushSync(() => {
    setCount(c => c + 1);
  });
  // 此时 DOM 已更新
  flushSync(() => {
    setFlag(f => !f);
  });
}
```

---

## 7. 多个 State 变量 vs 单个对象

### 7.1 多个独立的 State（推荐）

```jsx
function Form() {
  const [name, setName] = React.useState('');
  const [email, setEmail] = React.useState('');
  const [age, setAge] = React.useState(0);

  // 每个状态独立，更新互不影响
  // 只更新 name 时，email 和 age 相关的渲染被跳过
}
```

### 7.2 单个 State 对象

```jsx
function Form() {
  const [form, setForm] = React.useState({
    name: '',
    email: '',
    age: 0,
  });

  const updateField = (field, value) => {
    setForm(prev => ({ ...prev, [field]: value }));
  };

  return (
    <input
      value={form.name}
      onChange={(e) => updateField('name', e.target.value)}
    />
  );
}
```

### 7.3 对比与选择

| 场景 | 推荐方式 |
|------|---------|
| 状态之间独立 | 多个 useState |
| 状态之间紧密关联（如表单字段） | 单个对象 useState |
| 需要同时更新多个值 | 单个对象 useState |
| 需要精确控制重渲染范围 | 多个 useState |
| 状态较多（>5 个） | 考虑 useReducer |

### 7.4 使用 useReducer 管理复杂状态

当状态逻辑复杂、涉及多个子值、或下一个 state 依赖于上一个 state 时，`useReducer` 是更好的选择：

```jsx
function formReducer(state, action) {
  switch (action.type) {
    case 'UPDATE_FIELD':
      return { ...state, [action.field]: action.value };
    case 'RESET':
      return { name: '', email: '', age: 0 };
    case 'SET_ALL':
      return action.payload;
    default:
      return state;
  }
}

function Form() {
  const [form, dispatch] = React.useReducer(formReducer, {
    name: '',
    email: '',
    age: 0,
  });

  return (
    <input
      value={form.name}
      onChange={(e) => dispatch({
        type: 'UPDATE_FIELD',
        field: 'name',
        value: e.target.value,
      })}
    />
  );
}
```

---

## 8. State 更新是异步的

调用 `setState` 后，state 不会立即改变。React 会在下一次渲染时使用新值。

```jsx
function Counter() {
  const [count, setCount] = React.useState(0);

  const handleClick = () => {
    setCount(count + 1);
    console.log(count);  // 仍然是 0！state 还没更新
  };

  // 如果需要在 state 更新后执行操作，使用 useEffect
  React.useEffect(() => {
    console.log('count changed to:', count);
  }, [count]);
}
```

> 这不是 bug，而是 React 的设计——批处理优化、减少不必要的渲染。

---

## 9. 常见模式

### 9.1 布尔状态切换

```jsx
const [isOpen, setIsOpen] = React.useState(false);

// 简洁的切换方式
const toggle = () => setIsOpen(prev => !prev);
```

### 9.2 数组状态

```jsx
const [items, setItems] = React.useState([]);

// 添加
const addItem = (newItem) => setItems(prev => [...prev, newItem]);

// 删除
const removeItem = (id) => setItems(prev => prev.filter(item => item.id !== id));

// 更新
const updateItem = (id, updates) => {
  setItems(prev => prev.map(item =>
    item.id === id ? { ...item, ...updates } : item
  ));
};
```

### 9.3 对象状态的部分更新

```jsx
const [user, setUser] = React.useState({
  name: 'Alice',
  age: 25,
  address: { city: 'Beijing', zip: '100000' },
});

// 更新顶层字段
setUser(prev => ({ ...prev, name: 'Bob' }));

// 更新嵌套字段
setUser(prev => ({
  ...prev,
  address: { ...prev.address, city: 'Shanghai' },
}));
```

---

## 10. 常见面试问题

### Q1：useState 的初始值只在首次渲染时使用吗？

是的。后续渲染会忽略 `useState` 的初始值参数，React 使用内部存储的 state。但组件卸载再重新挂载时，会再次使用初始值。

### Q2：为什么不能在条件语句中调用 Hook？

React 依赖 Hook 的调用顺序来匹配 state 和对应的 `useState`。如果在条件语句中调用，顺序可能改变，导致 state 错配。

```jsx
// 错误
if (condition) {
  const [value, setValue] = React.useState('');  // 不要在条件中调用 Hook
}

// 正确
const [value, setValue] = React.useState('');
if (condition) {
  // 使用 value
}
```

### Q3：state 什么时候会丢失？

- 组件被卸载（从 DOM 中移除）
- 组件的 key 发生变化（React 认为是不同的组件）
- 父组件的相同位置渲染了不同类型的组件

### Q4：为什么连续调用两次 `setState(x + 1)` 只加了 1？

因为 `setState` 是异步的，两次调用中的 `x` 都是渲染时捕获的旧值。使用函数式更新 `setState(prev => prev + 1)` 可以正确累加。
