# JSX 表达式与语法

## 表达式嵌入基础

JSX 使用花括号 `{}` 将 JavaScript 表达式嵌入到模板中。这是 JSX 最核心的语法特性。

```jsx
const username = '张三'
const age = 25

function UserProfile() {
  return (
    <div>
      <p>姓名：{username}</p>
      <p>年龄：{age}</p>
      <p>明年：{age + 1}</p>
      <p>出生年份：{new Date().getFullYear() - age}</p>
    </div>
  )
}
```

> 关键区分：`{}` 内只能放**表达式**（有返回值的代码），不能放**语句**（if、for、switch 等）。

## 三元表达式

三元表达式是在 JSX 中实现条件逻辑的最常用方式：

```jsx
function Greeting({ isLoggedIn, username }) {
  return (
    <div>
      <h1>{isLoggedIn ? `欢迎回来，${username}！` : '请先登录'}</h1>
      <p>状态：{isLoggedIn ? '在线' : '离线'}</p>
    </div>
  )
}
```

### 嵌套三元表达式（不推荐）

```jsx
// 嵌套过深会导致可读性差
function StatusBadge({ status }) {
  return (
    <span>
      {status === 'active'
        ? '活跃'
        : status === 'inactive'
          ? '未激活'
          : status === 'banned'
            ? '已封禁'
            : '未知状态'}
    </span>
  )
}

// 更好的方式：提取为变量或使用对象映射
function StatusBadge({ status }) {
  const statusMap = {
    active: '活跃',
    inactive: '未激活',
    banned: '已封禁',
  }
  return <span>{statusMap[status] || '未知状态'}</span>
}
```

## 逻辑与（&&）短路运算

`&&` 用于"如果条件为真则渲染"的场景：

```jsx
function Notifications({ count, hasError, errorMessage }) {
  return (
    <div>
      {/* 仅当 count > 0 时显示角标 */}
      {count > 0 && <span className="badge">{count}</span>}

      {/* 仅在有错误时显示错误信息 */}
      {hasError && <p className="error">{errorMessage}</p>}

      {/* 组合条件 */}
      {count > 0 && count < 100 && (
        <span className="badge">{count}</span>
      )}
      {count >= 100 && <span className="badge">99+</span>}
    </div>
  )
}
```

### 注意：0 的陷阱

```jsx
function ProductList({ items }) {
  return (
    <div>
      {/* 危险！当 items.length 为 0 时，会渲染数字 0 而不是隐藏 */}
      {items.length && <ul>有商品</ul>}

      {/* 正确写法：显式转换为布尔值 */}
      {items.length > 0 && <ul>有商品</ul>}
      {!!items.length && <ul>有商品</ul>}
      {Boolean(items.length) && <ul>有商品</ul>}
    </div>
  )
}
```

> React 会渲染 `true`、`false`、`null`、`undefined`、数字等值。数字 `0` 是**会被渲染**的，所以 `0 && <Component/>` 会在页面上显示 `0`。

## 函数调用

可以在 JSX 中直接调用函数：

```jsx
function formatPrice(price) {
  return `¥${price.toFixed(2)}`
}

function getDiscountLevel(total) {
  if (total >= 1000) return '金牌会员'
  if (total >= 500) return '银牌会员'
  return '普通会员'
}

function OrderSummary({ items }) {
  const totalPrice = items.reduce((sum, item) => sum + item.price * item.qty, 0)

  return (
    <div>
      <p>总价：{formatPrice(totalPrice)}</p>
      <p>会员等级：{getDiscountLevel(totalPrice)}</p>
      <p>商品数量：{items.reduce((sum, item) => sum + item.qty, 0)} 件</p>
    </div>
  )
}
```

## 数组方法

### map 转换

`Array.map()` 是在 JSX 中渲染列表的标准方式：

```jsx
function FruitList() {
  const fruits = ['苹果', '香蕉', '橙子', '葡萄']

  return (
    <ul>
      {fruits.map((fruit) => (
        <li key={fruit}>{fruit}</li>
      ))}
    </ul>
  )
}
```

### filter 过滤

```jsx
function ActiveUsers({ users }) {
  return (
    <ul>
      {users
        .filter((user) => user.isActive)
        .map((user) => (
          <li key={user.id}>{user.name}</li>
        ))}
    </ul>
  )
}
```

### 链式调用

```jsx
function ProductStats({ products }) {
  const inStockCount = products.filter((p) => p.stock > 0).length
  const expensiveProducts = products
    .filter((p) => p.price > 100)
    .map((p) => p.name)

  return (
    <div>
      <p>在库商品：{inStockCount} 件</p>
      <p>
        高价商品：
        {expensiveProducts.length > 0
          ? expensiveProducts.join('、')
          : '无'}
      </p>
    </div>
  )
}
```

## 模板字面量

在 JSX 中使用模板字面量构建动态字符串：

```jsx
function UserCard({ user }) {
  const avatarUrl = `https://api.example.com/avatar/${user.id}`
  const fullName = `${user.firstName} ${user.lastName}`
  const memberSince = `注册于 ${user.createdAt.getFullYear()} 年`

  return (
    <div>
      <img src={avatarUrl} alt={`${fullName} 的头像`} />
      <h2>{fullName}</h2>
      <p>{memberSince}</p>
      <a href={`/user/${user.id}`}>查看 {fullName} 的主页</a>
    </div>
  )
}
```

## 可选链操作符（?.）

可选链在访问深层嵌套对象时非常有用：

```jsx
function UserDetail({ user }) {
  return (
    <div>
      {/* 安全访问嵌套属性 */}
      <p>姓名：{user?.name}</p>
      <p>城市：{user?.address?.city}</p>
      <p>街道：{user?.address?.street}</p>

      {/* 安全调用方法 */}
      <p>标签：{user?.getTags?.().join(', ')}</p>

      {/* 安全访问数组元素 */}
      <p>最新订单：{user?.orders?.[0]?.id}</p>
    </div>
  )
}
```

### 配合空值合并（??）

```jsx
function DisplayName({ user }) {
  return (
    <p>
      {/* 如果 nickname 为 null 或 undefined，使用 name */}
      昵称：{user?.nickname ?? user?.name ?? '匿名用户'}
    </p>
  )
}
```

## 解构赋值在 JSX 中的应用

```jsx
function UserCard({ user: { name, email, avatar }, isAdmin }) {
  return (
    <div className={isAdmin ? 'admin-card' : 'user-card'}>
      <img src={avatar} alt={name} />
      <h3>{name}</h3>
      <p>{email}</p>
    </div>
  )
}
```

## 对象展开传递属性

```jsx
function FormField({ label, ...inputProps }) {
  return (
    <div className="field">
      <label>{label}</label>
      <input {...inputProps} />
    </div>
  )
}

// 使用
function App() {
  return (
    <FormField
      label="用户名"
      type="text"
      name="username"
      placeholder="请输入用户名"
      required
    />
  )
}
```

## 小结

- `{}` 中只能放表达式，不能放语句
- 三元表达式是条件渲染的基础，避免过度嵌套
- `&&` 用于"条件为真则渲染"的场景，注意 `0` 的陷阱
- 数组方法（`map`、`filter`）配合使用实现列表渲染
- 可选链 `?.` 和空值合并 `??` 安全访问可能为 `null` 的属性
- 模板字面量构建动态字符串，解构赋值简化代码
