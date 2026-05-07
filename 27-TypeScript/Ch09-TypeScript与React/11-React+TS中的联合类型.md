# React+TS中的联合类型

## 一、概念说明

联合类型在 React 中常用于条件渲染、组件变体、异步状态等场景。通过 Discriminated Union（有标签联合），TypeScript 可以在条件分支中自动收窄类型，提供精确的类型提示和错误检查。

## 二、具体用法

### 2.1 条件渲染中的类型收窄

```tsx
type User = { name: string; email: string };
type Guest = { sessionId: string };

type AuthState =
  | { status: 'loading' }
  | { status: 'authenticated'; user: User }
  | { status: 'unauthenticated'; guest: Guest }
  | { status: 'error'; error: string };

function UserGreeting({ auth }: { auth: AuthState }) {
  // TypeScript 根据 status 自动收窄类型
  switch (auth.status) {
    case 'loading':
      return <div>加载中...</div>;

    case 'authenticated':
      // 这里 auth 被收窄为 { status: 'authenticated'; user: User }
      return <div>欢迎, {auth.user.name}</div>;

    case 'unauthenticated':
      return <div>访客: {auth.guest.sessionId}</div>;

    case 'error':
      return <div>错误: {auth.error}</div>;
  }
}
```

### 2.2 组件变体类型

```tsx
// 按钮变体 — 使用联合类型约束 props 组合
type ButtonProps =
  | {
      variant: 'primary' | 'secondary';
      children: React.ReactNode;
      onClick: () => void;
    }
  | {
      variant: 'link';
      href: string;
      children: React.ReactNode;
    }
  | {
      variant: 'icon';
      icon: React.ReactNode;
      onClick: () => void;
    };

function Button(props: ButtonProps) {
  switch (props.variant) {
    case 'primary':
    case 'secondary':
      return <button onClick={props.onClick}>{props.children}</button>;

    case 'link':
      return <a href={props.href}>{props.children}</a>;

    case 'icon':
      return <button onClick={props.onClick}>{props.icon}</button>;
  }
}
```

### 2.3 异步数据状态

```tsx
// 通用异步状态类型
type AsyncState<T> =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'success'; data: T }
  | { status: 'error'; error: Error };

function DataDisplay({ state }: { state: AsyncState<User[]> }) {
  if (state.status === 'loading') return <div>加载中...</div>;
  if (state.status === 'error') return <div>错误: {state.error.message}</div>;
  if (state.status === 'success') {
    return (
      <ul>
        {state.data.map(user => <li key={user.name}>{user.name}</li>)}
      </ul>
    );
  }
  return null;
}
```

### 2.4 Props 条件组合

```tsx
// 根据 type 属性决定其他必填 props
type ModalProps =
  | { type: 'confirm'; onConfirm: () => void; onCancel: () => void }
  | { type: 'alert'; onDismiss: () => void }
  | { type: 'custom'; children: React.ReactNode };

function Modal(props: ModalProps) {
  switch (props.type) {
    case 'confirm':
      return (
        <div>
          <button onClick={props.onConfirm}>确认</button>
          <button onClick={props.onCancel}>取消</button>
        </div>
      );
    case 'alert':
      return <button onClick={props.onDismiss}>知道了</button>;
    case 'custom':
      return <div>{props.children}</div>;
  }
}
```

## 三、注意事项与常见陷阱

1. **每个分支必须有唯一的判别属性**：`status`、`type` 等字面量类型
2. **使用 `never` 实现穷尽性检查**：确保所有分支都被处理
3. **不要过度嵌套联合类型**：保持类型结构扁平
4. **配合 `in` 操作符收窄类型**：`if ('user' in auth)` 检查属性存在
5. **条件渲染中先收窄再使用**：TypeScript 无法跨越 JSX 自动收窄
