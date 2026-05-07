# useState类型推断

## 一、概念说明

`useState` 是 React 中最基础的 Hook，TypeScript 会根据初始值自动推断状态类型。当初始值无法确定最终类型时（如 `null` 初始值），需要手动提供泛型参数。

**类型推断规则**：状态类型 = 初始值类型，更新函数的参数类型也基于此推断。

## 二、具体用法

### 2.1 自动类型推断

```tsx
import { useState } from 'react';

function AutoInfer() {
  // TypeScript 自动推断类型
  const [count, setCount] = useState(0);         // number
  const [name, setName] = useState('');           // string
  const [active, setActive] = useState(false);    // boolean
  const [items, setItems] = useState<string[]>([]); // 需要泛型：空数组推断为 never[]

  // setCount(123)      // OK — number
  // setCount('hello')  // 错误：不能将 string 赋给 number
  // setName(123)       // 错误：不能将 number 赋给 string
}
```

### 2.2 手动指定泛型

```tsx
// 场景一：初始值为 null，后续会有值
interface User {
  id: number;
  name: string;
  email: string;
}

function UserProfile() {
  // 必须用联合类型包含 null
  const [user, setUser] = useState<User | null>(null);

  const loadUser = async () => {
    const res = await fetch('/api/user');
    const data: User = await res.json();
    setUser(data); // OK
  };

  return (
    <div>
      {user ? <p>{user.name}</p> : <p>加载中...</p>}
    </div>
  );
}

// 场景二：空数组初始值
function TodoList() {
  const [todos, setTodos] = useState<string[]>([]);
  // 如果不指定泛型，todos 会被推断为 never[]

  setTodos(['学习 TypeScript']); // OK
}

// 场景三：多种可能的类型
function FormField() {
  const [value, setValue] = useState<string | number>('');
  setValue('hello');  // OK
  setValue(42);       // OK
}
```

### 2.3 复杂对象状态

```tsx
interface FormState {
  username: string;
  email: string;
  age: number;
  agree: boolean;
}

function RegistrationForm() {
  // 指定完整的初始值，TypeScript 自动推断类型
  const [form, setForm] = useState<FormState>({
    username: '',
    email: '',
    age: 0,
    agree: false,
  });

  // 部分更新 — 使用展开运算符
  const updateField = <K extends keyof FormState>(
    field: K,
    value: FormState[K]
  ) => {
    setForm(prev => ({ ...prev, [field]: value }));
  };

  return (
    <input
      value={form.username}
      onChange={e => updateField('username', e.target.value)}
    />
  );
}
```

### 2.4 函数式更新的类型

```tsx
function Counter() {
  const [count, setCount] = useState(0);

  // prev 自动推断为 number
  setCount(prev => prev + 1);

  // 如果需要异步更新，确保类型正确
  const incrementAsync = () => {
    setTimeout(() => {
      setCount(prev => prev + 1); // prev 始终是最新值
    }, 1000);
  };

  return <button onClick={incrementAsync}>+1</button>;
}
```

## 三、注意事项与常见陷阱

1. **空数组初始值**：`useState([])` 推断为 `never[]`，必须显式指定 `useState<string[]>([])`
2. **`null` 初始值**：必须用联合类型 `useState<User | null>(null)`，否则后续 `setUser` 无法接受非 null 值
3. **不要用 `as` 强制断言**：`useState(0 as number)` 是多余的，TypeScript 能自动推断
4. **对象状态更新**：始终使用展开运算符创建新对象，不要直接修改
5. **状态类型应尽量精确**：`useState<'loading' | 'success' | 'error'>('loading')` 比 `useState<string>('loading')` 更安全
