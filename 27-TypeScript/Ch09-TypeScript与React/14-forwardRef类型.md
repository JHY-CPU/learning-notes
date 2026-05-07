# forwardRef类型

## 一、概念说明

`React.forwardRef` 允许组件将 ref 传递给子组件的 DOM 元素或自定义组件。TypeScript 中需要为 forwardRef 提供两个泛型参数：**ref 的目标类型**和 **组件的 Props 类型**。

React 19 起 forwardRef 已不再是必需的，但理解其类型仍然重要。

## 二、具体用法

### 2.1 基本 forwardRef 类型

```tsx
import React, { forwardRef, useRef } from 'react';

// forwardRef 接受两个泛型参数：
// 第一个：ref 指向的元素类型
// 第二个：组件的 Props 类型
const CustomInput = forwardRef<HTMLInputElement, { label: string }>(
  function CustomInput({ label }, ref) {
    return (
      <div>
        <label>{label}</label>
        <input ref={ref} type="text" />
      </div>
    );
  }
);

// 父组件使用
function Parent() {
  const inputRef = useRef<HTMLInputElement>(null);

  const focusInput = () => {
    inputRef.current?.focus(); // 类型安全
  };

  return (
    <div>
      <CustomInput ref={inputRef} label="用户名" />
      <button onClick={focusInput}>聚焦</button>
    </div>
  );
}
```

### 2.2 透传额外属性

```tsx
// 扩展原生 input 属性
type InputProps = React.ComponentPropsWithRef<'input'> & {
  label: string;
  error?: string;
};

const FormInput = forwardRef<HTMLInputElement, InputProps>(
  function FormInput({ label, error, ...inputProps }, ref) {
    return (
      <div className="form-field">
        <label>{label}</label>
        <input ref={ref} {...inputProps} />
        {error && <span className="error">{error}</span>}
      </div>
    );
  }
);

// 使用 — 透传所有原生 input 属性
<FormInput
  ref={inputRef}
  label="邮箱"
  type="email"
  placeholder="请输入邮箱"
  required
/>;
```

### 2.3 多个 Ref 目标

```tsx
// 使用 useImperativeHandle 暴露自定义方法
import { forwardRef, useImperativeHandle, useRef } from 'react';

interface ModalHandle {
  open: () => void;
  close: () => void;
}

const Modal = forwardRef<ModalHandle, { title: string }>(
  function Modal({ title }, ref) {
    const dialogRef = useRef<HTMLDialogElement>(null);

    // 暴露给父组件的方法
    useImperativeHandle(ref, () => ({
      open: () => dialogRef.current?.showModal(),
      close: () => dialogRef.current?.close(),
    }));

    return (
      <dialog ref={dialogRef}>
        <h2>{title}</h2>
      </dialog>
    );
  }
);

// 父组件
function App() {
  const modalRef = useRef<ModalHandle>(null);
  return (
    <div>
      <button onClick={() => modalRef.current?.open()}>打开</button>
      <Modal ref={modalRef} title="提示" />
    </div>
  );
}
```

### 2.4 React 19 简化写法

```tsx
// React 19 中 ref 作为常规 prop，不需要 forwardRef
function MyInput({ label, ref }: { label: string; ref: React.Ref<HTMLInputElement> }) {
  return (
    <div>
      <label>{label}</label>
      <input ref={ref} />
    </div>
  );
}
```

## 三、注意事项与常见陷阱

1. **泛型顺序**：`forwardRef<RefType, PropsType>` — Ref 类型在前
2. **命名函数**：forwardRef 内使用命名函数便于 React DevTools 显示
3. **`useImperativeHandle` 的类型**：暴露的接口要显式定义
4. **React 19 不再需要 forwardRef**：ref 直接作为 props 传递
5. **不要在 forwardRef 中访问 children 的 ref**：ref 只能传给一个目标元素
