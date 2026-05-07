# JSX 中的条件渲染

## 概述

条件渲染是根据不同的状态或数据展示不同的 UI 内容。React 中有多种实现方式，适用于不同场景。

## if/else 方式

最直观的方式是将条件判断放在 return 之前：

```jsx
function StatusMessage({ status }) {
  if (status === 'loading') {
    return <p>加载中...</p>
  }

  if (status === 'error') {
    return <p>出错了，请稍后重试</p>
  }

  if (status === 'empty') {
    return <p>暂无数据</p>
  }

  return <p>加载成功</p>
}
```

这种方式适合**整个组件的渲染逻辑不同**的场景，代码清晰易读。

### 提前返回（Early Return）

提前返回是一种优雅的模式，避免深层嵌套：

```jsx
function UserProfile({ user }) {
  // 未登录
  if (!user) {
    return <p>请先登录</p>
  }

  // 账号被封禁
  if (user.isBanned) {
    return <p>您的账号已被封禁</p>
  }

  // 正常显示
  return (
    <div>
      <h2>{user.name}</h2>
      <p>邮箱：{user.email}</p>
    </div>
  )
}
```

## 三元表达式

三元表达式适合在 JSX 内部进行简单的二选一渲染：

```jsx
function LoginButton({ isLoggedIn, username }) {
  return (
    <div>
      {isLoggedIn ? (
        <div>
          <span>欢迎，{username}</span>
          <button>退出</button>
        </div>
      ) : (
        <button>登录</button>
      )}
    </div>
  )
}
```

### 使用技巧

```jsx
function Message({ type, content }) {
  return (
    <div className={`message message-${type}`}>
      {type === 'error' ? (
        <span className="icon-error">!</span>
      ) : type === 'success' ? (
        <span className="icon-success">✓</span>
      ) : (
        <span className="icon-info">i</span>
      )}
      <p>{content}</p>
    </div>
  )
}
```

## 逻辑与（&&）

`&&` 运算符用于"条件为真则渲染，否则渲染 null"的场景：

```jsx
function Header({ cartCount, isVip, hasNotification }) {
  return (
    <header>
      <h1>我的商城</h1>

      {/* 购物车角标：有商品时才显示 */}
      {cartCount > 0 && (
        <span className="cart-badge">{cartCount}</span>
      )}

      {/* VIP 标识 */}
      {isVip && <span className="vip-tag">VIP</span>}

      {/* 通知红点 */}
      {hasNotification && <span className="notification-dot" />}
    </header>
  )
}
```

### 多条件组合

```jsx
function AdminPanel({ user }) {
  return (
    <div>
      {/* 必须同时满足：已登录、是管理员、邮箱已验证 */}
      {user &&
        user.role === 'admin' &&
        user.emailVerified && (
          <div className="admin-panel">
            <h2>管理面板</h2>
            <button>管理用户</button>
            <button>系统设置</button>
          </div>
        )}
    </div>
  )
}
```

## 对象映射模式

当有多个分支条件时，使用对象映射比 if-else 或 switch 更优雅：

```jsx
function AlertMessage({ type }) {
  const alertConfig = {
    success: {
      className: 'alert-success',
      icon: '✓',
      message: '操作成功！',
    },
    error: {
      className: 'alert-error',
      icon: '✗',
      message: '操作失败，请重试。',
    },
    warning: {
      className: 'alert-warning',
      icon: '⚠',
      message: '请注意：此操作不可撤销。',
    },
    info: {
      className: 'alert-info',
      icon: 'ℹ',
      message: '这是一条提示信息。',
    },
  }

  const config = alertConfig[type] || alertConfig.info

  return (
    <div className={`alert ${config.className}`}>
      <span className="alert-icon">{config.icon}</span>
      <span>{config.message}</span>
    </div>
  )
}
```

### 配合组件映射

```jsx
// 将类型映射到组件
const iconComponents = {
  home: <HomeIcon />,
  settings: <SettingsIcon />,
  profile: <ProfileIcon />,
  logout: <LogoutIcon />,
}

function NavIcon({ type }) {
  return iconComponents[type] || <DefaultIcon />
}
```

## 守卫模式（Guard Pattern）

用于"仅在满足条件时渲染内容"的场景：

```jsx
function GuardedContent({ condition, fallback, children }) {
  if (!condition) {
    return fallback || null
  }
  return children
}

// 使用
function UserProfile({ user }) {
  return (
    <div>
      <GuardedContent
        condition={user?.isVip}
        fallback={<p>升级 VIP 解锁更多内容</p>}
      >
        <VipExclusiveContent />
      </GuardedContent>

      <GuardedContent condition={user?.isAdmin}>
        <AdminControls />
      </GuardedContent>
    </div>
  )
}
```

## 渲染行为详解

### null、undefined、false、true

这些值不会被渲染到 DOM 中：

```jsx
function Example() {
  return (
    <div>
      {null}        {/* 不渲染 */}
      {undefined}   {/* 不渲染 */}
      {false}       {/* 不渲染 */}
      {true}        {/* 不渲染 */}
    </div>
  )
}
```

### 数字 0

数字 `0` **会被渲染**到页面上，这是一个常见陷阱：

```jsx
function Cart({ items }) {
  return (
    <div>
      {/* 危险：items.length 为 0 时页面会显示 "0" */}
      {items.length && <p>有商品</p>}

      {/* 正确做法 */}
      {items.length > 0 && <p>有商品</p>}
      {items.length === 0 && <p>购物车为空</p>}
    </div>
  )
}
```

### 空字符串

空字符串 `''` 会被渲染（虽然页面上看不到，但 DOM 中存在文本节点）：

```jsx
function NameDisplay({ name }) {
  return (
    <p>
      {/* name 为空字符串时会产生空文本节点 */}
      姓名：{name || '未填写'}
    </p>
  )
}
```

## 条件样式

### 根据条件切换类名

```jsx
function Button({ variant, disabled, children }) {
  const className = [
    'btn',
    variant === 'primary' && 'btn-primary',
    variant === 'secondary' && 'btn-secondary',
    disabled && 'btn-disabled',
  ]
    .filter(Boolean)
    .join(' ')

  return <button className={className}>{children}</button>
}
```

### 使用 clsx / classnames 库

```jsx
import clsx from 'clsx'

function Button({ variant, size, disabled, children }) {
  return (
    <button
      className={clsx(
        'btn',
        {
          'btn-primary': variant === 'primary',
          'btn-secondary': variant === 'secondary',
          'btn-sm': size === 'small',
          'btn-lg': size === 'large',
          'btn-disabled': disabled,
        },
      )}
    >
      {children}
    </button>
  )
}
```

## 条件渲染的最佳实践

1. **提前返回**优于深层嵌套的三元表达式
2. **对象映射**优于多重 if-else 或 switch
3. **提取组件**而非在 JSX 中写复杂条件逻辑
4. **注意 0 的陷阱**：`count && <Component />` 当 count 为 0 时会渲染 0
5. **使用 `!!` 或 `Boolean()`** 将值显式转为布尔值

```jsx
// 提取组件前
function UserCard({ user }) {
  return (
    <div>
      {user ? (
        user.isActive ? (
          user.isVip ? (
            <VipBadge />
          ) : (
            <NormalBadge />
          )
        ) : (
          <InactiveBadge />
        )
      ) : (
        <GuestBadge />
      )}
    </div>
  )
}

// 提取组件后
function UserBadge({ user }) {
  if (!user) return <GuestBadge />
  if (!user.isActive) return <InactiveBadge />
  if (user.isVip) return <VipBadge />
  return <NormalBadge />
}

function UserCard({ user }) {
  return (
    <div>
      <UserBadge user={user} />
    </div>
  )
}
```

## 小结

- **提前返回**：适合整个组件渲染逻辑不同的场景
- **三元表达式**：适合简单的二选一场景
- **逻辑与 `&&`**：适合"有则渲染、无则跳过"的场景
- **对象映射**：适合多分支条件，比 switch 更灵活
- 注意 `0` 和空字符串的渲染行为差异
- 复杂条件逻辑应提取为独立组件或函数
