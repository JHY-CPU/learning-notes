# Props 基础

## 1. 什么是 Props

Props（properties 的缩写）是 React 组件之间传递数据的核心机制。父组件通过 props 将数据传递给子组件，实现**单向数据流**（从上到下）。

核心特性：**Props 是只读的**。子组件不能修改自己接收到的 props。

---

## 2. 传递 Props

### 2.1 基本传递

```jsx
// 父组件传递 props
function App() {
  return (
    <Greeting name="Alice" age={25} isStudent={true} />
  );
}

// 子组件接收 props
function Greeting(props) {
  return (
    <div>
      <p>Name: {props.name}</p>
      <p>Age: {props.age}</p>
      <p>Student: {props.isStudent ? 'Yes' : 'No'}</p>
    </div>
  );
}
```

### 2.2 传递不同类型的值

```jsx
function App() {
  const user = { name: 'Bob', email: 'bob@example.com' };
  const tags = ['react', 'javascript', 'frontend'];

  return (
    <UserProfile
      // 字符串
      title="Software Engineer"
      // 数字
      score={100}
      // 布尔值（简写形式）
      isActive
      // 对象
      user={user}
      // 数组
      tags={tags}
      // 函数
      onAction={() => console.log('clicked')}
      // JSX
      icon={<StarIcon />}
      // null / undefined
      badge={null}
    />
  );
}
```

> **注意**：
> - 字符串值使用引号：`name="Alice"`
> - 非字符串值使用花括号：`age={25}`
> - `isActive` 等价于 `isActive={true}`（布尔简写）

---

## 3. 读取 Props

### 3.1 通过 props 对象读取

```jsx
function UserCard(props) {
  return (
    <div>
      <h2>{props.name}</h2>
      <p>{props.bio}</p>
      <span>Followers: {props.followerCount}</span>
    </div>
  );
}
```

### 3.2 解构 Props（推荐）

```jsx
// 在参数位置解构
function UserCard({ name, bio, followerCount }) {
  return (
    <div>
      <h2>{name}</h2>
      <p>{bio}</p>
      <span>Followers: {followerCount}</span>
    </div>
  );
}

// 在函数体内部解构
function UserCard(props) {
  const { name, bio, followerCount } = props;
  return (
    <div>
      <h2>{name}</h2>
      <p>{bio}</p>
      <span>Followers: {followerCount}</span>
    </div>
  );
}
```

> 解构更清晰，减少了重复的 `props.` 前缀，是社区的主流写法。

---

## 4. Props 是只读的

这是 React 的核心规则之一：**组件不能修改自己的 props**。

```jsx
// 错误：尝试修改 props
function Counter(props) {
  props.count = props.count + 1; // TypeError! Props 是只读的
  return <div>{props.count}</div>;
}

// 正确：如果需要变化的数据，应该使用 state
function Counter({ initialCount }) {
  const [count, setCount] = React.useState(initialCount);
  return (
    <button onClick={() => setCount(count + 1)}>
      {count}
    </button>
  );
}
```

> **为什么 props 是只读的？**
> React 的哲学是 **单向数据流**。数据从父组件流向子组件，子组件通过回调函数通知父组件数据变化，由父组件决定是否更新。这种模式使数据流可预测、易于调试。

---

## 5. 默认 Props

### 5.1 默认参数值（推荐）

使用 ES6 函数参数默认值来定义 props 的默认值：

```jsx
function Button({ variant = 'primary', size = 'medium', disabled = false, children }) {
  return (
    <button
      className={`btn btn-${variant} btn-${size}`}
      disabled={disabled}
    >
      {children}
    </button>
  );
}

// 使用
<Button>Click me</Button>                         {/* variant='primary', size='medium' */}
<Button variant="danger" size="large">Delete</Button>  {/* 覆盖默认值 */}
```

### 5.2 对象类型的默认值

```jsx
function UserCard({ user = {}, style = {} }) {
  const { name = 'Anonymous', avatar = '/default-avatar.png' } = user;

  return (
    <div style={style}>
      <img src={avatar} alt={name} />
      <span>{name}</span>
    </div>
  );
}
```

### 5.3 函数类型的默认值

```jsx
function SearchInput({ onChange = () => {}, placeholder = 'Search...' }) {
  return <input onChange={onChange} placeholder={placeholder} />;
}

// 或者使用 noop 模式
const noop = () => {};
function SearchInput({ onChange = noop }) {
  return <input onChange={onChange} />;
}
```

### 5.4 旧式 `defaultProps`（已过时）

```jsx
// 旧写法，class 组件中仍可用，但不推荐在函数组件中使用
function TextInput({ type, placeholder }) {
  return <input type={type} placeholder={placeholder} />;
}

TextInput.defaultProps = {
  type: 'text',
  placeholder: 'Enter text...',
};
```

> **React 18.3+ 弃用警告**：`defaultProps` 在函数组件中已被标记为弃用。请使用参数默认值。

### 对比

| 特性 | 参数默认值 | `defaultProps` |
|------|-----------|---------------|
| 写法 | 函数参数中定义 | 函数外定义 |
| TypeScript 兼容 | 好 | 需额外类型声明 |
| React 18.3+ | 推荐 | 已弃用（函数组件） |
| 代码位置 | 紧邻组件定义 | 需要滚动到文件底部 |

---

## 6. Props 的类型

虽然 JavaScript 是动态类型的，但 React 社区广泛使用类型系统来约束 props。

### 6.1 PropTypes（运行时，已过时）

```jsx
import PropTypes from 'prop-types';

function UserCard({ name, age, isActive }) {
  return (
    <div>
      <h2>{name}</h2>
      <p>Age: {age}</p>
      <p>{isActive ? 'Online' : 'Offline'}</p>
    </div>
  );
}

UserCard.propTypes = {
  name: PropTypes.string.isRequired,
  age: PropTypes.number,
  isActive: PropTypes.bool,
};

UserCard.defaultProps = {
  age: 0,
  isActive: false,
};
```

> PropTypes 是运行时检查，仅在开发模式下生效。现代项目已基本被 TypeScript 取代。

### 6.2 TypeScript（推荐）

```tsx
interface UserCardProps {
  name: string;
  age?: number;           // 可选
  isActive?: boolean;     // 可选
  onAction?: () => void;  // 可选回调
}

function UserCard({ name, age = 0, isActive = false, onAction }: UserCardProps) {
  return (
    <div onClick={onAction}>
      <h2>{name}</h2>
      <p>Age: {age}</p>
      <p>{isActive ? 'Online' : 'Offline'}</p>
    </div>
  );
}
```

---

## 7. 展开 Props

### 7.1 传递时展开

当一个对象包含所有需要的 props 时，可以使用展开语法传递。

```jsx
function App() {
  const userProps = {
    name: 'Alice',
    age: 25,
    email: 'alice@example.com',
  };

  // 等价于 <UserCard name="Alice" age={25} email="alice@example.com" />
  return <UserCard {...userProps} />;
}

function UserCard({ name, age, email }) {
  return (
    <div>
      <p>{name}, {age}, {email}</p>
    </div>
  );
}
```

### 7.2 透传 Props（Forwarding）

将接收到的所有 props 传递给内部组件：

```jsx
function StyledInput(props) {
  // 将所有 props 透传给原生 input
  return <input className="styled-input" {...props} />;
}

// 使用
<StyledInput type="text" placeholder="Enter name" onChange={handleChange} />
```

### 7.3 展开的注意事项

```jsx
// 不推荐：盲目展开可能导致传递了不该传递的 props
function UserCard(props) {
  return <div {...props} />; // 可能将 className、style 等传到错误的位置
}

// 推荐：明确选择需要的 props
function UserCard({ name, age, ...rest }) {
  return <div {...rest}>{name}, {age}</div>;
}
```

---

## 8. Rest Props（...rest）

使用剩余参数语法收集未被解构的 props，常用于透传或组合场景。

```jsx
function Button({ variant, size, children, ...rest }) {
  // variant 和 size 被组件消费
  // rest 包含所有其他 props（如 onClick, disabled, className 等）
  const className = `btn btn-${variant} btn-${size}`;

  return (
    <button className={className} {...rest}>
      {children}
    </button>
  );
}

// 使用时
<Button
  variant="primary"
  size="large"
  onClick={handleClick}    // 进入 rest
  disabled={true}          // 进入 rest
  data-testid="submit"     // 进入 rest
>
  Submit
</Button>
```

### Rest Props 与 TypeScript

```tsx
interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant: 'primary' | 'secondary' | 'danger';
  size?: 'small' | 'medium' | 'large';
}

function Button({
  variant,
  size = 'medium',
  children,
  className,
  ...rest
}: ButtonProps) {
  const combinedClassName = `btn btn-${variant} btn-${size} ${className || ''}`;

  return (
    <button className={combinedClassName} {...rest}>
      {children}
    </button>
  );
}
```

---

## 9. Props 的常见使用模式

### 9.1 条件 Props

```jsx
function Alert({ type, children }) {
  const props = {
    className: `alert alert-${type}`,
    role: 'alert',
  };

  // 根据类型添加额外 props
  if (type === 'error') {
    props['aria-live'] = 'assertive';
  }

  return <div {...props}>{children}</div>;
}
```

### 9.2 Props 与 state 的区别

```jsx
// Props：从父组件接收，不可由自身修改
function Counter({ initialCount }) {
  // initialCount 是 props——由父组件决定
  // count 是 state——由组件自身管理
  const [count, setCount] = React.useState(initialCount);

  return (
    <button onClick={() => setCount(count + 1)}>
      Count: {count}
    </button>
  );
}
```

| 特性 | Props | State |
|------|-------|-------|
| 来源 | 父组件传递 | 组件内部定义 |
| 可变性 | 只读 | 可通过 `setState` 修改 |
| 作用 | 配置组件 | 管理内部数据 |
| 向下传递 | 通过 props 传递给子组件 | 可通过 props 传递给子组件 |

### 9.3 回调 Props

子组件通过调用父组件传入的函数来"通知"父组件：

```jsx
function TodoItem({ todo, onToggle, onDelete }) {
  return (
    <li>
      <input
        type="checkbox"
        checked={todo.completed}
        onChange={() => onToggle(todo.id)}  // 通知父组件
      />
      <span>{todo.text}</span>
      <button onClick={() => onDelete(todo.id)}>Delete</button>
    </li>
  );
}

// 父组件
function TodoApp() {
  const [todos, setTodos] = React.useState([]);

  const handleToggle = (id) => {
    setTodos(todos.map(t =>
      t.id === id ? { ...t, completed: !t.completed } : t
    ));
  };

  return (
    <TodoItem
      todo={todos[0]}
      onToggle={handleToggle}
      onDelete={(id) => setTodos(todos.filter(t => t.id !== id))}
    />
  );
}
```

---

## 10. 常见面试问题

### Q1：props 和 state 的区别是什么？

**Props** 是从父组件传递下来的数据，只读，子组件不能修改。**State** 是组件内部管理的数据，可变，通过 `setState` 修改。两者的变化都会触发重新渲染。

### Q2：为什么不能直接修改 props？

React 采用单向数据流设计。Props 的所有者是父组件，子组件直接修改 props 会导致数据来源不明确，破坏可预测性。如果子组件需要改变数据，应该通过回调函数通知父组件。

### Q3：如何给 props 设置默认值？

在函数组件中使用 ES6 默认参数：`function Button({ size = 'medium' }) {}`。`defaultProps` 在函数组件中已被弃用。

### Q4：展开 props `{...obj}` 有什么风险？

可能传递了组件不期望的属性（如事件处理器传给了错误的元素，或 `className` 被覆盖）。建议明确解构需要的 props，用 `...rest` 收集其余的。
