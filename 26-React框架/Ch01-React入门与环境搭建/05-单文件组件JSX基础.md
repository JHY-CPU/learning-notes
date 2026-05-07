# 单文件组件 JSX 基础

## 什么是 JSX

JSX（JavaScript XML）是 JavaScript 的语法扩展，允许在 JavaScript 中编写类似 HTML 的代码。它是 React 的核心语法。

```jsx
const element = <h1>Hello, React!</h1>
```

> JSX 不是字符串模板，它会在构建阶段被 Babel 编译为 `React.createElement()` 调用。

## JSX 基本语法规则

### 1. 必须有且只有一个根元素

```jsx
// 错误：多个根元素
function App() {
  return (
    <h1>标题</h1>
    <p>段落</p>
  )
}

// 正确：用一个 div 包裹
function App() {
  return (
    <div>
      <h1>标题</h1>
      <p>段落</p>
    </div>
  )
}

// 正确：使用 Fragment（不会产生多余的 DOM 节点）
function App() {
  return (
    <>
      <h1>标题</h1>
      <p>段落</p>
    </>
  )
}
```

### 2. 使用驼峰命名法

JSX 中的属性使用驼峰命名（camelCase），因为本质上是在写 JavaScript 对象属性：

```jsx
// HTML 写法 → JSX 写法
// class → className
<div className="container"></div>

// for → htmlFor
<label htmlFor="username">用户名</label>

// onclick → onClick
<button onClick={handleClick}>点击</button>

// tabindex → tabIndex
<div tabIndex={0}></div>

// onkeydown → onKeyDown
<input onKeyDown={handleKeyDown} />

// 自定义属性使用 data- 或 aria-
<div data-testid="my-component" aria-label="关闭"></div>
```

### 3. 自闭合标签必须写斜杠

```jsx
// 错误：HTML 中可以省略斜杠，JSX 中不行
<img src="logo.png">
<input type="text">

// 正确
<img src="logo.png" />
<input type="text" />
<br />
<hr />
```

### 4. 样式用对象传递

```jsx
// 内联样式是一个对象
<div style={{ color: 'red', fontSize: '16px' }}>
  红色文字
</div>

// 也可以先定义样式对象
const styles = {
  color: 'blue',
  backgroundColor: '#f0f0f0',
  padding: '10px',
}

function Card() {
  return <div style={styles}>卡片内容</div>
}
```

> CSS 属性名使用驼峰命名：`background-color` → `backgroundColor`，`font-size` → `fontSize`。

## 嵌入表达式

使用花括号 `{}` 在 JSX 中嵌入 JavaScript 表达式：

```jsx
const name = 'React'
const version = 19

function Info() {
  return (
    <div>
      <h1>欢迎学习 {name}</h1>
      <p>当前版本：{version}</p>
      <p>2 + 2 = {2 + 2}</p>
      <p>当前时间：{new Date().toLocaleString()}</p>
    </div>
  )
}
```

### 可以放在 `{}` 中的表达式

```jsx
function Expressions() {
  const user = { name: '张三', age: 25 }
  const fruits = ['苹果', '香蕉', '橙子']
  const getStatus = (code) => (code === 200 ? '成功' : '失败')

  return (
    <div>
      {/* 变量 */}
      <p>{user.name}</p>

      {/* 算术运算 */}
      <p>{user.age + 1}</p>

      {/* 三元表达式 */}
      <p>{user.age >= 18 ? '成年' : '未成年'}</p>

      {/* 函数调用 */}
      <p>{getStatus(200)}</p>

      {/* 数组方法 */}
      <ul>
        {fruits.map((f) => (
          <li key={f}>{f}</li>
        ))}
      </ul>

      {/* 逻辑与 */}
      {user.age >= 18 && <span>可以投票</span>}
    </div>
  )
}
```

### 不能放在 `{}` 中的语句

```jsx
// 错误：不能放语句（if、for、switch 等）
function Wrong() {
  return (
    <div>
      {/* 以下都是错误的 */}
      {if (true) { ... }}
      {for (let i = 0; i < 5; i++) { ... }}
      {const x = 1}
    </div>
  )
}

// 正确：用表达式替代
function Correct() {
  const items = [1, 2, 3, 4, 5]
  return (
    <div>
      {true ? '是' : '否'}
      {items.map((n) => (
        <span key={n}>{n}</span>
      ))}
    </div>
  )
}
```

## Fragment（片段）

Fragment 允许将多个子元素分组，而不在 DOM 中添加额外节点：

```jsx
// 短语法（常用）
function List() {
  return (
    <>
      <li>项目一</li>
      <li>项目二</li>
      <li>项目三</li>
    </>
  )
}

// 完整语法（需要传递 key 时使用）
function List({ items }) {
  return (
    <React.Fragment>
      {items.map((item) => (
        <React.Fragment key={item.id}>
          <h3>{item.title}</h3>
          <p>{item.content}</p>
        </React.Fragment>
      ))}
    </React.Fragment>
  )
}
```

> Fragment 不会在 DOM 中产生额外的节点，保持 HTML 结构干净。

## JSX 的编译过程

JSX 并不是浏览器能直接理解的语法，需要经过编译：

```
JSX 源码 → Babel 编译 → React.createElement() → 虚拟DOM → 真实DOM
```

### 编译示例

```jsx
// JSX 源码
const element = (
  <div className="greeting">
    <h1>Hello, {name}!</h1>
  </div>
)
```

```js
// Babel 编译结果
const element = React.createElement(
  'div',
  { className: 'greeting' },
  React.createElement('h1', null, 'Hello, ', name, '!'),
)
```

```js
// React 17+ 新的 JSX 转换（无需手动导入 React）
// 由 @babel/plugin-transform-react-jsx 自动处理
import { jsx as _jsx } from 'react/jsx-runtime'

const element = _jsx('div', {
  className: 'greeting',
  children: _jsx('h1', { children: ['Hello, ', name, '!'] }),
})
```

> React 17 以后，JSX 编译不再需要 `import React from 'react'`，编译器会自动引入运行时。

### React.createElement 参数

```js
React.createElement(type, props, ...children)
```

| 参数 | 说明 |
|------|------|
| `type` | 元素类型（标签名字符串或组件函数） |
| `props` | 属性对象（包括事件处理函数） |
| `children` | 子元素（可以有多个） |

## 注释

JSX 中的注释写法：

```jsx
function Comment() {
  return (
    <div>
      {/* 这是 JSX 中的单行注释 */}
      <h1>Hello</h1>

      {/* 
        这是 JSX 中的
        多行注释 
      */}
    </div>
  )
}
```

在 JSX 外部使用标准 JavaScript 注释：

```jsx
function App() {
  // 这是标准的 JS 单行注释
  const name = 'React' /* 这是标准的 JS 行内注释 */

  return <div>{name}</div>
}
```

## 条件属性

当属性值为 `true` 时可以简写：

```jsx
// 等价写法
<input type="text" disabled={true} />
<input type="text" disabled />

// 等价写法
<MyComponent isAuth={true} />
<MyComponent isAuth />
```

## 小结

- JSX 是 JavaScript 的语法扩展，最终会被编译为 `React.createElement` 调用
- 必须有一个根元素，可以使用 Fragment 避免多余 DOM 节点
- 使用 `{}` 嵌入表达式，不能使用语句
- 属性名使用驼峰命名（`className`、`onClick`）
- 自闭合标签必须写斜杠 `/>`
- React 17+ 不再需要手动导入 React 来使用 JSX
