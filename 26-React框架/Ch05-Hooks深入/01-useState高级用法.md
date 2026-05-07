# useState 高级用法

## 目录

1. [基础回顾](#基础回顾)
2. [惰性初始化（Lazy Initialization）](#惰性初始化)
3. [状态更新函数模式（Updater Function）](#状态更新函数模式)
4. [批量更新行为（Batching）](#批量更新行为)
5. [使用 useReducer 替代复杂状态](#使用-usereducer-替代复杂状态)
6. [useState 与对象状态（常见陷阱）](#usestate-与对象状态)
7. [使用 key 属性重置状态](#使用-key-属性重置状态)

---

## 基础回顾

```jsx
import { useState } from 'react';

function Counter() {
  const [count, setCount] = useState(0);

  return (
    <button onClick={() => setCount(count + 1)}>
      当前计数: {count}
    </button>
  );
}
```

---

## 惰性初始化

当初始状态的计算成本较高时，可以传递一个**函数**给 `useState`，该函数只在**组件首次渲染时**调用一次。

### 为什么需要惰性初始化？

```jsx
// ❌ 不好：每次渲染都会调用 expensiveComputation()
const [state, setState] = useState(expensiveComputation());

// ✅ 好：只在首次渲染时调用
const [state, setState] = useState(() => expensiveComputation());
```

### 实际场景

```jsx
function App() {
  // 从 localStorage 读取值 — 涉及 I/O 操作，不应每次渲染都执行
  const [user, setUser] = useState(() => {
    const saved = localStorage.getItem('user');
    return saved ? JSON.parse(saved) : { name: '访客' };
  });

  // 生成随机种子 — 只需要一次
  const [sessionId] = useState(() => crypto.randomUUID());

  // 复杂的初始数据处理
  const [data, setData] = useState(() => {
    const raw = fetchInitialData(); // 同步获取
    return raw.map(item => transformItem(item));
  });

  return <div>欢迎, {user.name}</div>;
}
```

### 惰性初始化的执行时机

```
组件首次挂载
    │
    ▼
调用初始化函数 ← 只有这时调用一次
    │
    ▼
返回初始值 → 设置 state
    │
    ▼
组件重新渲染（setState触发）
    │
    ▼
初始化函数 不再调用 ✗
```

---

## 状态更新函数模式

`setState` 不仅可以接收一个新值，还可以接收一个**函数**（称为 updater function），该函数接收前一个状态作为参数，返回新的状态。

### 基本语法

```jsx
// 直接赋值
setCount(count + 1);

// 函数式更新（推荐在依赖前一状态时使用）
setCount(prevCount => prevCount + 1);
```

### 为什么函数式更新更安全？

```jsx
function Counter() {
  const [count, setCount] = useState(0);

  // ❌ 问题：连续调用多次时，count 是同一个闭包值
  const handleClickBad = () => {
    setCount(count + 1); // count = 0, 设置为 1
    setCount(count + 1); // count = 0, 仍然设置为 1
    setCount(count + 1); // count = 0, 仍然设置为 1
    // 最终结果: 1, 而不是 3!
  };

  // ✅ 正确：函数式更新确保基于最新值
  const handleClickGood = () => {
    setCount(prev => prev + 1); // prev = 0, 返回 1
    setCount(prev => prev + 1); // prev = 1, 返回 2
    setCount(prev => prev + 1); // prev = 2, 返回 3
    // 最终结果: 3 ✓
  };

  return (
    <div>
      <p>计数: {count}</p>
      <button onClick={handleClickBad}>错误方式 (+1)</button>
      <button onClick={handleClickGood}>正确方式 (+3)</button>
    </div>
  );
}
```

### 对象状态的函数式更新

```jsx
function UserProfile() {
  const [user, setUser] = useState({ name: '张三', age: 25 });

  const updateName = (newName) => {
    setUser(prev => ({ ...prev, name: newName }));
  };

  const incrementAge = () => {
    setUser(prev => ({ ...prev, age: prev.age + 1 }));
  };

  return (
    <div>
      <p>{user.name}, {user.age}岁</p>
      <button onClick={() => updateName('李四')}>改名</button>
      <button onClick={incrementAge}>长一岁</button>
    </div>
  );
}
```

### 数组状态的函数式更新

```jsx
function TodoList() {
  const [todos, setTodos] = useState([]);

  const addTodo = (text) => {
    setTodos(prev => [...prev, { id: Date.now(), text, done: false }]);
  };

  const removeTodo = (id) => {
    setTodos(prev => prev.filter(todo => todo.id !== id));
  };

  const toggleTodo = (id) => {
    setTodos(prev =>
      prev.map(todo =>
        todo.id === id ? { ...todo, done: !todo.done } : todo
      )
    );
  };

  // ... 渲染
}
```

---

## 批量更新行为

React 会将多个 `setState` 调用合并为一次重新渲染，这称为**批量更新（Batching）**。

### React 17 中的批量更新

在 React 17 中，批量更新**仅在 React 事件处理函数**中生效：

```jsx
// React 17 行为
function Counter() {
  const [count, setCount] = useState(0);
  const [flag, setFlag] = useState(false);

  // ✅ React 事件处理：批量更新（1次渲染）
  const handleClick = () => {
    setCount(c => c + 1);
    setFlag(f => !f);
    // 只触发 1 次重新渲染
  };

  // ❌ setTimeout / Promise / 原生事件：不会批量（2次渲染）
  const handleAsync = () => {
    setTimeout(() => {
      setCount(c => c + 1); // 触发 1 次渲染
      setFlag(f => !f);     // 再触发 1 次渲染
    }, 1000);
  };

  return (
    <div>
      <p>Count: {count}, Flag: {String(flag)}</p>
      <button onClick={handleClick}>React 事件（批量）</button>
      <button onClick={handleAsync}>异步（不批量）</button>
    </div>
  );
}
```

### React 18+ 中的自动批量更新

React 18 引入了**自动批量更新**，无论状态更新发生在何处，都会被自动合并：

```jsx
// React 18+ 行为
function Counter() {
  const [count, setCount] = useState(0);
  const [flag, setFlag] = useState(false);

  // ✅ 所有情况都自动批量（1次渲染）
  const handleAsync = () => {
    setTimeout(() => {
      setCount(c => c + 1);  // ┐
      setFlag(f => !f);      // ┘ 合并为 1 次渲染
    }, 1000);
  };

  // ✅ Promise 中也自动批量
  const handleFetch = async () => {
    const data = await fetchData();
    setCount(data.count);   // ┐
    setFlag(data.flag);     // ┘ 合并为 1 次渲染
  };

  // ✅ 原生事件中也自动批量
  const handleNative = () => {
    const el = document.getElementById('btn');
    el.addEventListener('click', () => {
      setCount(c => c + 1);  // ┐
      setFlag(f => !f);      // ┘ 合并为 1 次渲染
    });
  };

  return (
    <div>
      <p>Count: {count}, Flag: {String(flag)}</p>
      <button onClick={handleAsync}>异步更新（React 18 自动批量）</button>
    </div>
  );
}
```

### 批量更新对照表

| 场景 | React 17 | React 18+ |
|------|----------|-----------|
| React 事件处理函数 | 批量 | 批量 |
| setTimeout / setInterval | **不批量** | 批量 |
| Promise 回调 | **不批量** | 批量 |
| 原生事件监听器 | **不批量** | 批量 |
| await 之后 | **不批量** | 批量 |

### 强制同步刷新（flushSync）

如果确实需要立即触发渲染，可以使用 `flushSync`：

```jsx
import { flushSync } from 'react-dom';

function App() {
  const [count, setCount] = useState(0);

  const handleClick = () => {
    flushSync(() => {
      setCount(c => c + 1); // 立即同步更新
    });
    // 此时 count 已经是最新值
    console.log(count); // 输出新值

    setCount(c => c + 1); // 这一次仍然会批量
  };

  return <button onClick={handleClick}>计数: {count}</button>;
}
```

> **提示**：`flushSync` 会损害性能，应谨慎使用。常见的使用场景是：更新状态后需要立即操作 DOM 或读取布局信息。

---

## 使用 useReducer 替代复杂状态

当状态逻辑复杂、涉及多个子值或依赖前一状态时，`useReducer` 通常是更好的选择。

### useState 的局限

```jsx
// ❌ 多个相关状态散落，逻辑分散
function ShoppingCart() {
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [discount, setDiscount] = useState(0);

  const addItem = (item) => {
    setItems(prev => [...prev, item]);
    setTotal(prev => prev + item.price);
    // discount 逻辑也可能需要更新...
  };

  // 每次操作都需要同步维护多个状态
}
```

### useReducer 的优势

```jsx
// ✅ 状态和逻辑集中管理
const initialState = { items: [], total: 0, discount: 0 };

function cartReducer(state, action) {
  switch (action.type) {
    case 'ADD_ITEM':
      const newItems = [...state.items, action.payload];
      const newTotal = newItems.reduce((sum, item) => sum + item.price, 0);
      return {
        ...state,
        items: newItems,
        total: newTotal,
        discount: newTotal > 200 ? newTotal * 0.1 : 0,
      };
    case 'REMOVE_ITEM':
      const filteredItems = state.items.filter(
        item => item.id !== action.payload
      );
      const updatedTotal = filteredItems.reduce(
        (sum, item) => sum + item.price, 0
      );
      return {
        ...state,
        items: filteredItems,
        total: updatedTotal,
        discount: updatedTotal > 200 ? updatedTotal * 0.1 : 0,
      };
    case 'CLEAR':
      return initialState;
    default:
      throw new Error(`未知操作: ${action.type}`);
  }
}

function ShoppingCart() {
  const [state, dispatch] = useReducer(cartReducer, initialState);

  return (
    <div>
      <p>购物车 ({state.items.length} 件商品)</p>
      <p>总计: ¥{state.total - state.discount}</p>
      {state.discount > 0 && <p>已优惠: ¥{state.discount}</p>}
      <button onClick={() => dispatch({
        type: 'ADD_ITEM',
        payload: { id: Date.now(), name: '商品', price: 99 }
      })}>
        添加商品
      </button>
    </div>
  );
}
```

### useState vs useReducer 选择指南

| 场景 | 推荐 |
|------|------|
| 简单的布尔值、数字、字符串 | `useState` |
| 状态之间相互依赖 | `useReducer` |
| 状态逻辑复杂，有多个操作类型 | `useReducer` |
| 下一个状态依赖前一个状态 | 两者均可（useReducer 更清晰） |
| 需要将状态逻辑传递给子组件 | `useReducer`（dispatch 引用稳定） |

---

## useState 与对象状态

### 常见陷阱：忘记展开旧状态

```jsx
const [user, setUser] = useState({ name: '张三', age: 25, city: '北京' });

// ❌ 错误：丢失了 age 和 city
setUser({ name: '李四' });

// ✅ 正确：展开旧状态再覆盖
setUser(prev => ({ ...prev, name: '李四' }));
```

### 常见陷阱：深层嵌套对象

```jsx
const [user, setUser] = useState({
  name: '张三',
  address: {
    province: '广东省',
    city: '深圳市',
    detail: {
      district: '南山区',
      street: '科技园路',
    },
  },
});

// ❌ 浅拷贝无法更新嵌套属性
setUser(prev => ({
  ...prev,
  address: { ...prev.address, city: '广州市' },
  // detail 仍然指向旧对象！
}));

// ✅ 正确：逐层展开
setUser(prev => ({
  ...prev,
  address: {
    ...prev.address,
    city: '广州市',
    detail: {
      ...prev.address.detail,
      district: '天河区', // 同时更新深层属性
    },
  },
}));
```

### 推荐：使用 Immer 简化深层更新

```bash
npm install immer
```

```jsx
import { produce } from 'immer';

const [user, setUser] = useState({
  name: '张三',
  address: { city: '深圳', detail: { district: '南山' } },
});

// ✅ 使用 Immer：可以"直接修改"（实际上是不可变更新）
setUser(prev =>
  produce(prev, draft => {
    draft.address.city = '广州';
    draft.address.detail.district = '天河';
  })
);
```

### 常见陷阱：直接修改状态对象

```jsx
const [items, setItems] = useState([{ id: 1, name: '苹果' }]);

// ❌ 错误：直接修改状态！React 不会检测到变化
items.push({ id: 2, name: '香蕉' });
setItems(items); // 引用相同，React 不会重新渲染

// ✅ 正确：创建新数组
setItems([...items, { id: 2, name: '香蕉' }]);

// ✅ 或使用 concat（返回新数组）
setItems(prev => prev.concat({ id: 2, name: '香蕉' }));
```

### 常见陷阱：对象引用比较

```jsx
function App() {
  const [config, setConfig] = useState({ theme: 'dark' });

  // ❌ 问题：每次渲染都创建新对象，导致依赖 config 的子组件不必要地重新渲染
  return <Child config={{ theme: 'dark' }} />;

  // ✅ 正确：复用同一个对象引用
  return <Child config={config} />;
}
```

---

## 使用 key 属性重置状态

当 `key` 发生变化时，React 会**销毁旧组件并创建新组件实例**，所有状态都会重置为初始值。

### 基本用法

```jsx
function ProfileEditor({ userId }) {
  const [name, setName] = useState('');
  const [bio, setBio] = useState('');

  // ❌ 问题：切换用户时，name 和 bio 不会重置
  // 因为组件实例没变，useState 只在首次渲染时设置初始值
  return (
    <div>
      <input value={name} onChange={e => setName(e.target.value)} />
      <textarea value={bio} onChange={e => setBio(e.target.value)} />
    </div>
  );
}

// ✅ 解决方案：使用 key
function ProfileEditorWrapper({ userId }) {
  return <ProfileEditor key={userId} userId={userId} />;
}

function ProfileEditor({ userId }) {
  const [name, setName] = useState('');
  const [bio, setBio] = useState('');
  // key 变化时，整个组件重新挂载，状态全部重置

  return (
    <div>
      <h3>编辑用户 {userId}</h3>
      <input value={name} onChange={e => setName(e.target.value)} />
      <textarea value={bio} onChange={e => setBio(e.target.value)} />
    </div>
  );
}
```

### 实际场景：表单重置

```jsx
function ContactForm() {
  const [formKey, setFormKey] = useState(0);
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = (data) => {
    saveData(data);
    setSubmitted(true);
    setFormKey(prev => prev + 1); // 重置表单
  };

  return (
    <div>
      {submitted && <p className="success">提交成功！</p>}
      <InnerForm key={formKey} onSubmit={handleSubmit} />
    </div>
  );
}

function InnerForm({ onSubmit }) {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [message, setMessage] = useState('');

  return (
    <form onSubmit={(e) => { e.preventDefault(); onSubmit({ name, email, message }); }}>
      <input value={name} onChange={e => setName(e.target.value)} placeholder="姓名" />
      <input value={email} onChange={e => setEmail(e.target.value)} placeholder="邮箱" />
      <textarea value={message} onChange={e => setMessage(e.target.value)} placeholder="留言" />
      <button type="submit">提交</button>
    </form>
  );
}
```

### 实际场景：Tab 切换重置编辑器

```jsx
function TabbedEditor() {
  const [activeTab, setActiveTab] = useState('tab1');

  return (
    <div>
      <nav>
        <button onClick={() => setActiveTab('tab1')}>文档 1</button>
        <button onClick={() => setActiveTab('tab2')}>文档 2</button>
        <button onClick={() => setActiveTab('tab3')}>文档 3</button>
      </nav>
      {/* 每个 tab 切换时，编辑器状态重置 */}
      <RichTextEditor key={activeTab} documentId={activeTab} />
    </div>
  );
}
```

---

## 总结与最佳实践

1. **惰性初始化**：初始值计算昂贵时，传函数 `useState(() => computeInitial())`
2. **函数式更新**：连续更新依赖前一状态时，使用 `setState(prev => newValue)`
3. **React 18 批量更新**：自动合并所有状态更新，减少不必要的渲染
4. **复杂状态用 useReducer**：状态之间有依赖关系或操作类型多时
5. **对象/数组状态**：永远创建新引用，不要直接修改，考虑使用 Immer
6. **key 重置**：需要重置组件全部状态时，改变 `key` 值
7. **避免过度拆分**：相关的状态放在一起，无关的状态分开

```jsx
// ✅ 相关状态合并
const [position, setPosition] = useState({ x: 0, y: 0 });

// ✅ 无关状态分开
const [name, setName] = useState('');
const [age, setAge] = useState(0);
```
