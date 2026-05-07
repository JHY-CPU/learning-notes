# Ch12-01 React 渲染机制

## 目录

1. [Virtual DOM 与 Diff 算法](#1-virtual-dom-与-diff-算法)
2. [Reconciliation 协调过程](#2-reconciliation-协调过程)
3. [Fiber 架构基础](#3-fiber-架构基础)
4. [Render 阶段与 Commit 阶段](#4-render-阶段与-commit-阶段)
5. [Keys 与协调](#5-keys-与协调)
6. [React 为什么重新渲染](#6-react-为什么重新渲染)
7. [Trigger vs Render vs Commit](#7-trigger-vs-render-vs-commit)

---

## 1. Virtual DOM 与 Diff 算法

### 1.1 什么是 Virtual DOM

Virtual DOM 是对真实 DOM 的轻量级 JavaScript 表示。它是一个普通的 JS 对象树，描述了 UI 应该长什么样：

```jsx
// JSX
<div className="container">
  <h1>Hello</h1>
  <p>World</p>
</div>

// 编译后的 Virtual DOM (简化表示)
{
  type: "div",
  props: { className: "container" },
  children: [
    { type: "h1", props: {}, children: ["Hello"] },
    { type: "p", props: {}, children: ["World"] }
  ]
}
```

### 1.2 为什么需要 Virtual DOM

```
直接操作 DOM 的问题：
  1. DOM 操作昂贵 - 每次操作都可能触发重排 (reflow) 和重绘 (repaint)
  2. 手动优化困难 - 很难知道哪些 DOM 节点需要更新
  3. 批量更新困难 - 频繁的 DOM 操作会导致性能问题

Virtual DOM 的解决方案：
  1. 在 JS 层面做计算 - 比直接操作 DOM 快得多
  2. Diff 算法找出最小变更 - 只更新真正变化的部分
  3. 批量更新 - 收集所有变更后一次性应用到真实 DOM
```

### 1.3 Diff 算法的规则

React 的 Diff 算法基于两个假设（启发式规则），将 O(n^3) 的复杂度优化到 O(n)：

```js
// 规则一：不同类型的元素产生不同的树
// 如果根元素类型不同，React 会销毁整棵子树重建
<div>        →  <span>       // 整棵子树被销毁重建
  <p>A</p>      <p>A</p>
</div>            </span>

// 规则二：通过 key 属性标识稳定同级元素
// 使用 key 可以帮助 React 识别哪些元素发生了变化
<ul>
  {items.map(item => (
    <li key={item.id}>{item.text}</li>  // key 帮助 React 跟踪元素
  ))}
</ul>
```

### 1.4 Diff 算法的具体过程

```
步骤 1：比较根元素类型
  - 类型不同 → 销毁旧树，创建新树
  - 类型相同 → 比较属性变化，递归子节点

步骤 2：比较同级子节点
  - 传统方案：逐个比较 → O(n^3) 复杂度
  - React 方案：使用 key 标识 → O(n) 复杂度

步骤 3：列表 Diff
  - 有 key 时：通过 key 匹配，识别移动/增删
  - 无 key 时：按索引逐个比较（可能导致错误复用）
```

---

## 2. Reconciliation 协调过程

### 2.1 什么是 Reconciliation

Reconciliation（协调）是 React 确定需要对真实 DOM 做哪些更改的过程：

```
新的 JSX 树 ←→ 旧的 Fiber 树
     ↓
  比较 (Diffing)
     ↓
  生成新的 Fiber 树
     ↓
  标记变更 (Placement, Update, Deletion)
     ↓
  提交到真实 DOM
```

### 2.2 Reconciliation 的触发条件

```jsx
// 以下情况会触发 Reconciliation：

// 1. setState 调用
function Counter() {
  const [count, setCount] = useState(0);
  // 点击按钮触发 Reconciliation
  return <button onClick={() => setCount(c => c + 1)}>{count}</button>;
}

// 2. 父组件重新渲染
function Parent() {
  const [value, setValue] = useState(0);
  // 每次 Parent 重新渲染，Child 也会重新渲染
  return <Child value={value} />;
}

// 3. Context 值变化
function Consumer() {
  const theme = useContext(ThemeContext);
  // theme 变化时，所有 Consumer 都会重新渲染
  return <div style={{ color: theme.color }}>Hello</div>;
}

// 4. forceUpdate（类组件）
class LegacyComponent extends React.Component {
  handleClick() {
    this.forceUpdate(); // 强制重新渲染
  }
}
```

### 2.3 Reconciliation 过程详解

```
完整流程：
  1. 触发更新 (setState / useState / props 变化)
  2. 创建新的 React Element 树
  3. 与当前 Fiber 树进行比较
  4. 生成新的 Fiber 树（包含变更标记）
  5. 收集所有变更（称为 "effect list"）
  6. 将变更应用到真实 DOM
  7. 执行 useEffect 等副作用
```

---

## 3. Fiber 架构基础

### 3.1 什么是 Fiber

Fiber 是 React 16 引入的新架构，是 Virtual DOM 节点的增强表示。每个 React 组件对应一个 Fiber 节点：

```js
// Fiber 节点结构 (简化)
{
  // 组件类型
  type: "div",          // 或组件函数/类
  key: "my-key",

  // 树结构关系
  child: Fiber | null,  // 第一个子节点
  sibling: Fiber | null, // 下一个兄弟节点
  return: Fiber | null,  // 父节点

  // 待处理的工作
  pendingProps: {},      // 新的 props
  memoizedProps: {},     // 上次渲染使用的 props
  memoizedState: {},     // 上次渲染使用的 state

  // 副作用标记
  effectTag: "Update",  // Placement | Update | Deletion | ...
  nextEffect: Fiber,     // 副作用链表

  // 调度相关
  lanes: 0,              // 优先级（React 18 使用 lanes 模型）
  alternate: Fiber,      // 双缓冲：指向另一棵树中的对应节点
}
```

### 3.2 为什么需要 Fiber

```
Stack Reconciliation (React 15) 的问题：
  - 递归遍历组件树，一次性完成所有工作
  - 无法中断，无法暂停
  - 大组件树更新时会阻塞主线程
  - 导致动画卡顿、输入延迟等

Fiber Reconciliation (React 16+) 的优势：
  - 链表结构，可以随时中断和恢复
  - 优先级调度，高优先级更新可以打断低优先级
  - 增量渲染，将大任务拆分为小任务
  - 更好的并发能力
```

### 3.3 Fiber 树结构

```
传统树结构（递归遍历）：        Fiber 链表结构（可中断）：

       A                              A
      / \                           / | \
     B   C        →               B -- C
    / \                           |    |
   D   E                         D -- E

递归：A → B → D → E → C         链表：A → B → D → E → C
（无法中断）                    （每个节点都是独立的工作单元，可中断）
```

### 3.4 双缓冲 (Double Buffering)

React 使用两棵 Fiber 树实现双缓冲：

```
current 树：当前显示在屏幕上的 UI 对应的 Fiber 树
workInProgress 树：正在构建的新 Fiber 树

更新过程：
  1. 基于 current 树创建 workInProgress 树
  2. 在 workInProgress 树上执行更新
  3. 更新完成后，workInProgress 树变成新的 current 树
  4. 旧的 current 树变为备用，下次更新时复用

current ←→ workInProgress
  ↓              ↓
 屏幕上        正在构建
  显示的        新的 UI
```

---

## 4. Render 阶段与 Commit 阶段

### 4.1 两阶段模型

React 的更新过程分为两个阶段：

```
Render 阶段（可中断）：          Commit 阶段（不可中断）：
  - 调用组件函数                  - 应用 DOM 变更
  - 计算变更                     - 执行生命周期方法
  - 创建 Fiber 树                 - 执行 useLayoutEffect
  - 标记副作用                   - 提交副作用到 DOM
     ↓                              ↓
  可以被高优先级任务中断          必须同步完成
```

### 4.2 Render 阶段

```
Render 阶段的工作：
  1. 遍历 Fiber 树
  2. 对每个 Fiber 节点：
     a. 调用组件函数（函数组件）或 render 方法（类组件）
     b. 比较新旧 props/state
     c. 决定是否需要更新子树
     d. 协调子节点（对比新旧 children）
     e. 标记副作用（placement, update, deletion）
  3. 构建 effect list（所有需要处理的副作用链表）

特点：
  - 可以中断（时间切片）
  - 可以被高优先级更新打断
  - 不会产生用户可见的变化
  - 可能被调用多次（Strict Mode）
```

### 4.3 Commit 阶段

```
Commit 阶段的三个子阶段：

  1. Before Mutation（DOM 变更前）
     - 读取 DOM 状态
     - 执行 getSnapshotBeforeUpdate

  2. Mutation（DOM 变更）
     - 执行 DOM 操作（插入、更新、删除）
     - 执行 componentWillUnmount

  3. Layout（DOM 变更后）
     - 执行 componentDidMount / componentDidUpdate
     - 执行 useLayoutEffect
     - 更新 refs

特点：
  - 不可中断
  - 同步执行
  - 产生用户可见的变化
  - 速度很快（只做 DOM 操作）
```

### 4.4 时间切片 (Time Scheduling)

```
传统渲染：                     时间切片渲染：

[========= 渲染任务 =========]  [==1ms==][==1ms==][==1ms==]
                              16ms 内                          60fps

长任务阻塞主线程：               每执行 1ms 就让出主线程：
  - 动画卡顿                    - 动画流畅
  - 输入无响应                   - 输入响应
  - 用户体验差                   - 用户体验好

原理：
  - 利用 requestIdleCallback 或 MessageChannel
  - 将大任务拆分为小工作单元
  - 每个 Fiber 节点是一个工作单元
  - 每帧给 React 约 5ms，其余时间留给浏览器
```

---

## 5. Keys 与协调

### 5.1 Key 的作用

Key 是 React 用来识别列表中元素身份的特殊属性：

```jsx
// 没有 key - 按索引比较
// 问题：插入元素后，后续所有元素都会被错误地认为"修改了"
<ul>
  <li>A</li>    // index 0
  <li>B</li>    // index 1
  <li>C</li>    // index 2
</ul>
// 在开头插入 D 后：
<ul>
  <li>D</li>    // index 0 - React 认为原来的 A 变成了 D → 更新
  <li>A</li>    // index 1 - React 认为原来的 B 变成了 A → 更新
  <li>B</li>    // index 2 - React 认为原来的 C 变成了 B → 更新
  <li>C</li>    // index 3 - 新增
</ul>

// 有 key - 按 key 比较
// 正确识别元素身份，只做最小变更
<ul>
  <li key="a">A</li>
  <li key="b">B</li>
  <li key="c">C</li>
</ul>
// 在开头插入 D 后：
<ul>
  <li key="d">D</li>   // 新增
  <li key="a">A</li>   // 不变
  <li key="b">B</li>   // 不变
  <li key="c">C</li>   // 不变
</ul>
```

### 5.2 Key 的最佳实践

```jsx
// ✅ 好的 key - 稳定且唯一
{users.map(user => (
  <UserCard key={user.id} user={user} />
))}

// ❌ 坏的 key - 使用索引
{users.map((user, index) => (
  <UserCard key={index} user={user} />
))}

// ❌ 坏的 key - 随机值
{users.map(user => (
  <UserCard key={Math.random()} user={user} />
))}

// ✅ 复合 key - 当需要多个维度标识时
{items.map(item => (
  <Item key={`${item.type}-${item.id}`} item={item} />
))}
```

### 5.3 索引作为 Key 的危险场景

```jsx
// 场景：可排序/可过滤的列表
function SortableList() {
  const [users, setUsers] = useState([
    { id: 1, name: "Alice", score: 90 },
    { id: 2, name: "Bob", score: 85 },
    { id: 3, name: "Charlie", score: 95 },
  ]);

  const sortByScore = () => {
    setUsers((prev) => [...prev].sort((a, b) => b.score - a.score));
  };

  return (
    <div>
      <button onClick={sortByScore}>按分数排序</button>
      <ul>
        {/* ❌ 使用索引作为 key */}
        {users.map((user, index) => (
          <li key={index}>
            <UserInput defaultValue={user.name} />
          </li>
        ))}
      </ul>
    </div>
  );
}

// 排序后，index 0 对应的 UserInput 还是原来的实例
// 但数据已经变了，导致输入框显示错误的值
// 正确做法：key={user.id}
```

---

## 6. React 为什么重新渲染

### 6.1 触发重新渲染的原因

```jsx
// 1. useState 的 setter 被调用
const [count, setCount] = useState(0);
setCount(1); // 触发重新渲染

// 2. useReducer 的 dispatch 被调用
const [state, dispatch] = useReducer(reducer, initialState);
dispatch({ type: "INCREMENT" }); // 触发重新渲染

// 3. 父组件重新渲染
function Parent() {
  const [value, setValue] = useState(0);
  return <Child />; // Parent 重新渲染时，Child 也会重新渲染
}

// 4. Context 值变化
function Consumer() {
  const theme = useContext(ThemeContext); // theme 变化时重新渲染
  return <div>{theme.color}</div>;
}

// 5. Props 变化
function Child({ value }) {
  return <div>{value}</div>; // value 变化时重新渲染
}

// 6. 强制更新（类组件）
this.forceUpdate(); // 强制重新渲染
```

### 6.2 不会触发重新渲染的情况

```jsx
// 1. 设置相同的值
const [count, setCount] = useState(0);
setCount(0); // React 会跳过重新渲染（Object.is 比较）

// 2. 在事件处理中设置状态（React 会批处理）
function handleClick() {
  setCount(c => c + 1);
  setName("New Name");
  setActive(true);
  // 只触发一次重新渲染（React 18 自动批处理）
}

// 3. useRef 的 current 变化
const ref = useRef(0);
ref.current = 100; // 不会触发重新渲染
```

### 6.3 React 18 批处理

```jsx
// React 17：只有 React 事件处理器中有批处理
// React 18：所有情况下都有自动批处理

function App() {
  const [count, setCount] = useState(0);
  const [flag, setFlag] = useState(false);

  // React 17：2次渲染；React 18：1次渲染
  async function handleClick() {
    await fetchData();
    setCount((c) => c + 1); // Promise 回调中
    setFlag((f) => !f);     // React 17 会分别渲染，18 会批处理
  }

  return <button onClick={handleClick}>Click</button>;
}

// 如果不需要批处理，使用 flushSync
import { flushSync } from "react-dom";
flushSync(() => {
  setCount((c) => c + 1);
}); // 强制同步渲染
flushSync(() => {
  setFlag((f) => !f);
}); // 再次同步渲染
```

---

## 7. Trigger vs Render vs Commit

### 7.1 三个阶段详解

```
1. Trigger（触发）
   └─ 什么触发了更新？
      - setState / useState 的 setter
      - useReducer 的 dispatch
      - 父组件重新渲染
      - forceUpdate

2. Render（渲染）
   └─ React 做了什么？
      - 调用组件函数获取新的 JSX
      - 对比新旧 Fiber 树（Reconciliation）
      - 标记需要变更的节点
      - ⚠️ 这是可中断的阶段

3. Commit（提交）
   └─ 浏览器做了什么？
      - React 将变更应用到真实 DOM
      - 执行同步生命周期方法
      - 执行 useLayoutEffect
      - ⚠️ 这是不可中断的阶段

4. Browser Paint（绘制）
   └─ 浏览器将 DOM 变更绘制到屏幕
   └─ useEffect 在此之后执行
```

### 7.2 时间线

```
时间线：
|------Trigger------|----Render (可中断)----|--Commit--|--Paint--|--useEffect--|

                   ↑                         ↑          ↑          ↑
              setState()                DOM 更新    浏览器绘制   useEffect
```

### 7.3 完整示例

```jsx
function Counter() {
  const [count, setCount] = useState(0);

  console.log("1. Render: 组件函数被调用");

  useEffect(() => {
    console.log("4. useEffect: 在浏览器绘制后执行");
  });

  useLayoutEffect(() => {
    console.log("3. useLayoutEffect: DOM 更新后、绘制前执行");
  });

  const handleClick = () => {
    console.log("0. Trigger: setState 被调用");
    setCount((c) => c + 1);
    console.log("0.5 Trigger 结束: 但渲染还没开始");
  };

  console.log("1.5 Render 阶段: JSX 被计算");

  return (
    <div>
      <span>{count}</span>
      <button onClick={handleClick}>+1</button>
    </div>
  );
}

// 点击按钮后的输出顺序：
// "0. Trigger: setState 被调用"
// "0.5 Trigger 结束: 但渲染还没开始"
// "1. Render: 组件函数被调用"
// "1.5 Render 阶段: JSX 被计算"
// "3. useLayoutEffect: DOM 更新后、绘制前执行"
// --- 浏览器绘制 ---
// "4. useEffect: 在浏览器绘制后执行"
```

---

## 小结

- Virtual DOM 是对真实 DOM 的 JS 表示，React 通过 Diff 算法找出最小变更
- Fiber 架构将递归遍历改为可中断的链表遍历，支持并发渲染
- React 更新分为 Render（可中断）和 Commit（不可中断）两个阶段
- Key 帮助 React 正确识别列表中元素的身份，应使用稳定且唯一的值
- React 18 的自动批处理减少了不必要的重新渲染次数
- 理解 Trigger/Render/Commit 的区别有助于编写更高效的 React 代码
