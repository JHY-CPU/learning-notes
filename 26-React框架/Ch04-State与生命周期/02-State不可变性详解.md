# State 不可变性详解

## 1. 为什么不可变性至关重要

React 依赖**引用比较**来决定是否需要重新渲染。直接修改对象或数组时，引用不变，React 无法感知数据已经变化。

### 1.1 引用相等性（Referential Equality）

```jsx
const obj1 = { name: 'Alice' };
const obj2 = obj1;           // 同一个引用
const obj3 = { ...obj1 };    // 新引用

obj1 === obj2;  // true  —— 同一个对象
obj1 === obj3;  // false —— 内容相同但引用不同
```

### 1.2 React 的协调机制

React 使用浅比较来判断 state 是否变化：

```jsx
function App() {
  const [items, setItems] = React.useState([1, 2, 3]);

  const handleBadUpdate = () => {
    items.push(4);         // 修改了同一个数组
    setItems(items);       // 引用没变！React 认为 state 没变，不重渲染
  };

  const handleGoodUpdate = () => {
    setItems([...items, 4]); // 新数组，新引用，触发重渲染
  };
}
```

### 1.3 性能优化的基础

`React.memo`、`useMemo`、`useCallback` 等性能优化工具都依赖引用相等性判断：

```jsx
const MemoizedChild = React.memo(function Child({ user }) {
  console.log('Child rendered');
  return <div>{user.name}</div>;
});

function Parent() {
  const [count, setCount] = React.useState(0);
  const [name, setName] = React.useState('Alice');

  // 错误：每次渲染都创建新对象，memo 失效
  const user = { name };

  // 正确：使用 useMemo 缓存
  const user = React.useMemo(() => ({ name }), [name]);

  return (
    <>
      <MemoizedChild user={user} />
      <button onClick={() => setCount(count + 1)}>Count: {count}</button>
    </>
  );
}
```

---

## 2. 更新原始值（Primitives）

原始类型（string、number、boolean、null、undefined）本身就是不可变的，直接赋新值即可。

```jsx
const [count, setCount] = React.useState(0);
const [name, setName] = React.useState('');
const [isActive, setIsActive] = React.useState(false);

// 直接传入新值
setCount(42);
setName('React');
setIsActive(true);

// 基于旧值更新
setCount(prev => prev + 1);
setName(prev => prev.toUpperCase());
setIsActive(prev => !prev);
```

---

## 3. 更新数组

**核心原则**：永远不用 `push`、`pop`、`splice`、`sort`、`reverse` 等会修改原数组的方法。使用返回新数组的方法。

### 3.1 添加元素

```jsx
const [items, setItems] = React.useState(['a', 'b', 'c']);

// 添加到末尾
setItems(prev => [...prev, 'd']);            // ['a', 'b', 'c', 'd']

// 添加到开头
setItems(prev => ['z', ...prev]);            // ['z', 'a', 'b', 'c']

// 添加到指定位置
const insertAt = (arr, index, newItem) => [
  ...arr.slice(0, index),
  newItem,
  ...arr.slice(index),
];
setItems(prev => insertAt(prev, 1, 'x'));    // ['a', 'x', 'b', 'c']
```

### 3.2 删除元素

```jsx
const [items, setItems] = React.useState([
  { id: 1, name: 'Alice' },
  { id: 2, name: 'Bob' },
  { id: 3, name: 'Carol' },
]);

// 按索引删除
setItems(prev => prev.filter((_, index) => index !== 1));

// 按条件删除
setItems(prev => prev.filter(item => item.id !== 2));

// 删除第一个
setItems(prev => prev.slice(1));

// 删除最后一个
setItems(prev => prev.slice(0, -1));
```

### 3.3 更新元素

```jsx
const [items, setItems] = React.useState([
  { id: 1, name: 'Alice', done: false },
  { id: 2, name: 'Bob', done: false },
]);

// 按索引更新
setItems(prev => prev.map((item, index) =>
  index === 1 ? { ...item, name: 'Robert' } : item
));

// 按条件更新（最常用）
setItems(prev => prev.map(item =>
  item.id === 2 ? { ...item, done: true } : item
));

// 更新所有元素
setItems(prev => prev.map(item => ({ ...item, done: true })));
```

### 3.4 排序

```jsx
const [numbers, setNumbers] = React.useState([3, 1, 4, 1, 5]);

// 错误：sort 会修改原数组
// numbers.sort();  // 修改了原数组引用
// setNumbers(numbers);

// 正确方式一：先复制再排序
setNumbers(prev => [...prev].sort((a, b) => a - b));

// 正确方式二：使用 toSorted（ES2023+，返回新数组）
setNumbers(prev => prev.toSorted((a, b) => a - b));

// 降序排列
setNumbers(prev => [...prev].sort((a, b) => b - a));
```

### 3.5 反转

```jsx
const [letters, setLetters] = React.useState(['a', 'b', 'c']);

// 错误：reverse 修改原数组
// letters.reverse();

// 正确方式一：先复制再反转
setLetters(prev => [...prev].reverse());

// 正确方式二：使用 toReversed（ES2023+）
setLetters(prev => prev.toReversed());
```

### 3.6 splice 替代方案

`splice` 会修改原数组，需要替代方案：

```jsx
const [items, setItems] = React.useState(['a', 'b', 'c', 'd']);

// splice(start, deleteCount) —— 删除
// 使用 toSpliced（ES2023+）
setItems(prev => prev.toSpliced(1, 2));        // 删除索引 1 开始的 2 个

// 或手动实现
const removeAt = (arr, start, count = 1) => [
  ...arr.slice(0, start),
  ...arr.slice(start + count),
];
setItems(prev => removeAt(prev, 1, 2));        // ['a', 'd']

// splice(start, 0, item) —— 插入
setItems(prev => prev.toSpliced(1, 0, 'x'));   // 插入 'x' 到索引 1

// splice(start, 1, item) —— 替换
setItems(prev => prev.toSpliced(1, 1, 'x'));   // 将索引 1 替换为 'x'
```

### 3.7 移动元素

```jsx
const moveItem = (arr, fromIndex, toIndex) => {
  const newArr = [...arr];
  const [item] = newArr.splice(fromIndex, 1);  // 移除
  newArr.splice(toIndex, 0, item);              // 插入
  return newArr;
};

// 更简洁的 immutable 写法
const moveItem = (arr, fromIndex, toIndex) => {
  const item = arr[fromIndex];
  return arr
    .filter((_, i) => i !== fromIndex)
    .toSpliced(toIndex, 0, item);
};
```

### 3.8 数组操作速查表

| 操作 | 修改原数组（避免） | 不可变替代 |
|------|-------------------|-----------|
| 添加末尾 | `push` | `[...arr, item]` |
| 添加开头 | `unshift` | `[item, ...arr]` |
| 删除末尾 | `pop` | `arr.slice(0, -1)` |
| 删除开头 | `shift` | `arr.slice(1)` |
| 删除/插入/替换 | `splice` | `arr.toSpliced()` 或 `slice` + 展开 |
| 排序 | `sort` | `[...arr].sort()` 或 `arr.toSorted()` |
| 反转 | `reverse` | `[...arr].reverse()` 或 `arr.toReversed()` |
| 更新元素 | 直接赋值 | `arr.map()` |

---

## 4. 更新对象

### 4.1 更新顶层属性

```jsx
const [user, setUser] = React.useState({
  name: 'Alice',
  age: 25,
  email: 'alice@example.com',
});

// 更新单个属性
setUser(prev => ({ ...prev, name: 'Bob' }));

// 更新多个属性
setUser(prev => ({
  ...prev,
  name: 'Bob',
  age: 26,
}));

// Object.assign 方式（不常用）
setUser(prev => Object.assign({}, prev, { name: 'Bob' }));
```

### 4.2 更新嵌套对象

```jsx
const [user, setUser] = React.useState({
  name: 'Alice',
  address: {
    city: 'Beijing',
    zip: '100000',
    geo: {
      lat: 39.9,
      lng: 116.4,
    },
  },
});

// 更新嵌套属性：需要逐层展开
setUser(prev => ({
  ...prev,
  address: {
    ...prev.address,
    city: 'Shanghai',
  },
}));

// 更新更深层的嵌套
setUser(prev => ({
  ...prev,
  address: {
    ...prev.address,
    geo: {
      ...prev.address.geo,
      lat: 31.2,
    },
  },
}));
```

### 4.3 添加和删除属性

```jsx
const [config, setConfig] = React.useState({ theme: 'dark', lang: 'zh' });

// 添加属性
setConfig(prev => ({ ...prev, fontSize: 16 }));

// 删除属性
const { lang, ...rest } = config;
setConfig(rest);  // { theme: 'dark' }

// 动态删除
const deleteKey = (obj, key) => {
  const { [key]: _, ...rest } = obj;
  return rest;
};
setConfig(prev => deleteKey(prev, 'lang'));
```

---

## 5. Immer 简介

当嵌套层级深时，手动展开非常繁琐。**Immer** 让你可以用"看起来像修改"的语法，实际上生成不可变更新。

### 5.1 安装

```bash
npm install immer
```

### 5.2 基本用法

```jsx
import { produce } from 'immer';

const [user, setUser] = React.useState({
  name: 'Alice',
  address: { city: 'Beijing', geo: { lat: 39.9, lng: 116.4 } },
});

// 没有 Immer：繁琐
setUser(prev => ({
  ...prev,
  address: {
    ...prev.address,
    geo: {
      ...prev.address.geo,
      lat: 31.2,
    },
  },
}));

// 有 Immer：直观
setUser(prev =>
  produce(prev, (draft) => {
    draft.address.geo.lat = 31.2;  // 看起来像直接修改，实际是安全的
  })
);
```

### 5.3 与 useState 结合

```jsx
// 封装一个 immer 版本的 setState
function useImmer(initialValue) {
  const [state, setState] = React.useState(initialValue);
  const setImmerState = React.useCallback((updater) => {
    setState(typeof updater === 'function'
      ? produce(updater)
      : produce(() => updater)
    );
  }, []);
  return [state, setImmerState];
}

// 使用
const [todos, setTodos] = useImmer([
  { id: 1, text: 'Learn React', done: false },
]);

setTodos((draft) => {
  draft[0].done = true;  // 直接修改，Immer 保证不可变性
});
```

### 5.4 Immer 的原理

Immer 使用 **Proxy** 拦截对 draft 对象的修改操作，自动创建一个新的不可变对象。你"修改"的其实是一个 Proxy 包装的草稿，Immer 记录所有变化，最后生成新的 state。

---

## 6. 常见错误及修正

### 6.1 数组直接修改

```jsx
// 错误
items.push(newItem);
items[0] = updatedItem;
items.sort();

// 正确
setItems(prev => [...prev, newItem]);
setItems(prev => prev.map((item, i) => i === 0 ? updatedItem : item));
setItems(prev => [...prev].sort());
```

### 6.2 对象直接修改

```jsx
// 错误
user.name = 'Bob';
user.address.city = 'Shanghai';

// 正确
setUser(prev => ({ ...prev, name: 'Bob' }));
setUser(prev => ({ ...prev, address: { ...prev.address, city: 'Shanghai' } }));
```

### 6.3 混淆浅拷贝和深拷贝

```jsx
// 展开运算符只做浅拷贝
const oldState = { a: { b: 1 } };
const newState = { ...oldState };

newState.a === oldState.a;  // true! 内层对象仍是同一个引用

// 修改内层会污染旧数据
newState.a.b = 2;
console.log(oldState.a.b);  // 2 —— oldState 也被修改了！

// 正确：逐层展开
const newState = { ...oldState, a: { ...oldState.a, b: 2 } };
```

### 6.4 在 useEffect 中读取更新后的 state

```jsx
const [count, setCount] = React.useState(0);

// 错误：count 可能不是最新的
const handleClick = () => {
  setCount(count + 1);
  saveToServer(count);  // 用的是旧的 count
};

// 正确：在 useEffect 中处理
React.useEffect(() => {
  saveToServer(count);
}, [count]);

// 或使用函数式更新获取最新值
const handleClick = () => {
  setCount(prev => {
    const newCount = prev + 1;
    saveToServer(newCount);
    return newCount;
  });
};
```

---

## 7. 总结

| 数据类型 | 不可变更新方式 |
|---------|--------------|
| 原始值 | 直接传新值 |
| 数组 | `spread`, `filter`, `map`, `concat`, `slice`, `toSpliced`/`toSorted`/`toReversed` |
| 对象 | `spread`, 解构赋值 |
| 深层嵌套 | 逐层展开 或 使用 Immer |

**核心记忆**：更新 state 就是创建新值替换旧值，永远不要修改原来的值。
