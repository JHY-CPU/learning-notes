# forwardRef 与 useImperativeHandle

在 React 中，默认情况下父组件无法直接获取子组件内部的 DOM 节点或方法。`forwardRef` 和 `useImperativeHandle` 提供了向父组件暴露子组件内部能力的机制。

---

## 一、forwardRef 基础

### 1.1 问题：ref 无法直接传递

```jsx
// ❌ ref 不是普通 props，无法这样传递
function MyInput({ ref, ...props }) {
  return <input ref={ref} {...props} />;
}

// 使用时 ref 不会指向 input 元素
function App() {
  const inputRef = useRef();
  return <MyInput ref={inputRef} />;  // inputRef.current 为 undefined
}
```

### 1.2 forwardRef 解决方案

```jsx
import { forwardRef } from 'react';

// forwardRef 包裹组件，接收 props 和 ref 两个参数
const MyInput = forwardRef(function MyInput(props, ref) {
  return <input ref={ref} {...props} />;
});

// 也可以使用箭头函数
const MyInput2 = forwardRef((props, ref) => {
  return <input ref={ref} {...props} />;
});

// 现在 ref 可以正确指向 input DOM
function App() {
  const inputRef = useRef();

  useEffect(() => {
    inputRef.current.focus();  // ✅ 正确指向 input DOM
  }, []);

  return <MyInput ref={inputRef} placeholder="输入内容" />;
}
```

### 1.3 forwardRef 的 TypeScript 类型

```tsx
import { forwardRef, InputHTMLAttributes } from 'react';

// 方式 1：显式类型
const MyInput = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(
  function MyInput(props, ref) {
    return <input ref={ref} {...props} />;
  }
);

// 方式 2：使用 ComponentPropsWithoutRef（推荐）
import { ComponentPropsWithoutRef } from 'react';

const MyInput = forwardRef<HTMLInputElement, ComponentPropsWithoutRef<'input'>>(
  function MyInput(props, ref) {
    return <input ref={ref} {...props} />;
  }
);

// 方式 3：带自定义 props
interface MyInputProps extends ComponentPropsWithoutRef<'input'> {
  label: string;
  error?: string;
}

const MyInput = forwardRef<HTMLInputElement, MyInputProps>(
  function MyInput({ label, error, ...props }, ref) {
    return (
      <div>
        <label>{label}</label>
        <input ref={ref} {...props} />
        {error && <span className="error">{error}</span>}
      </div>
    );
  }
);
```

---

## 二、useImperativeHandle

`useImperativeHandle` 允许你自定义暴露给父组件的 ref 值，而不是直接暴露 DOM 节点。

### 2.1 基础用法

```jsx
import { forwardRef, useImperativeHandle, useRef } from 'react';

const FancyInput = forwardRef(function FancyInput(props, ref) {
  const inputRef = useRef(null);

  // 自定义暴露给父组件的 ref 值
  useImperativeHandle(ref, () => ({
    // 只暴露 focus 方法，不暴露整个 DOM
    focus: () => {
      inputRef.current.focus();
    },
    // 可以暴露任意方法
    clear: () => {
      inputRef.current.value = '';
    },
    getValue: () => {
      return inputRef.current.value;
    },
  }));

  return <input ref={inputRef} {...props} />;
});

// 使用
function App() {
  const inputRef = useRef();

  return (
    <div>
      <FancyInput ref={inputRef} />
      <button onClick={() => inputRef.current.focus()}>聚焦</button>
      <button onClick={() => inputRef.current.clear()}>清空</button>
      <button onClick={() => alert(inputRef.current.getValue())}>
        获取值
      </button>
    </div>
  );
}
```

### 2.2 为什么需要 useImperativeHandle

```jsx
// 不用 useImperativeHandle：父组件拿到完整 DOM
const BadInput = forwardRef(function BadInput(props, ref) {
  return <input ref={ref} {...props} />;
});

function App() {
  const ref = useRef();
  // 父组件可以做任何 DOM 操作，包括破坏子组件封装
  ref.current.style.display = 'none';  // 危险
  ref.current.parentNode.removeChild(ref.current);  // 非常危险
}

// 用 useImperativeHandle：父组件只能访问指定的方法
const GoodInput = forwardRef(function GoodInput(props, ref) {
  const inputRef = useRef(null);

  useImperativeHandle(ref, () => ({
    focus: () => inputRef.current.focus(),
    // 不暴露 style、parentNode 等 DOM 属性
  }));

  return <input ref={inputRef} {...props} />;
});
```

### 2.3 常见应用场景

#### 表单验证方法

```jsx
const FormInput = forwardRef(function FormInput({ label, required, validate }, ref) {
  const inputRef = useRef(null);
  const [error, setError] = useState('');

  useImperativeHandle(ref, () => ({
    // 验证方法
    validate: () => {
      const value = inputRef.current.value;

      if (required && !value.trim()) {
        setError(`${label}不能为空`);
        return false;
      }

      if (validate) {
        const errorMsg = validate(value);
        if (errorMsg) {
          setError(errorMsg);
          return false;
        }
      }

      setError('');
      return true;
    },

    // 获取值
    getValue: () => inputRef.current.value,

    // 重置
    reset: () => {
      inputRef.current.value = '';
      setError('');
    },

    // 聚焦
    focus: () => inputRef.current.focus(),
  }));

  return (
    <div>
      <label>{label}</label>
      <input ref={inputRef} />
      {error && <span className="error">{error}</span>}
    </div>
  );
});

// 父组件使用
function RegistrationForm() {
  const nameRef = useRef();
  const emailRef = useRef();
  const passwordRef = useRef();

  const handleSubmit = (e) => {
    e.preventDefault();
    const refs = [nameRef, emailRef, passwordRef];
    const allValid = refs.every(ref => ref.current.validate());

    if (allValid) {
      // 提交表单
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <FormInput ref={nameRef} label="姓名" required />
      <FormInput ref={emailRef} label="邮箱" required validate={(v) =>
        !v.includes('@') ? '邮箱格式不正确' : null
      } />
      <FormInput ref={passwordRef} label="密码" required validate={(v) =>
        v.length < 8 ? '密码至少8位' : null
      } />
      <button type="submit">注册</button>
    </form>
  );
}
```

#### 滚动控制

```jsx
const ScrollableList = forwardRef(function ScrollableList({ items }, ref) {
  const containerRef = useRef(null);
  const itemRefs = useRef({});

  useImperativeHandle(ref, () => ({
    scrollToTop: () => {
      containerRef.current.scrollTo({ top: 0, behavior: 'smooth' });
    },
    scrollToBottom: () => {
      containerRef.current.scrollTo({
        top: containerRef.current.scrollHeight,
        behavior: 'smooth',
      });
    },
    scrollToItem: (id) => {
      itemRefs.current[id]?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    },
  }));

  return (
    <div ref={containerRef} style={{ height: 400, overflow: 'auto' }}>
      {items.map(item => (
        <div key={item.id} ref={el => itemRefs.current[item.id] = el}>
          {item.text}
        </div>
      ))}
    </div>
  );
});
```

---

## 三、React 19 中的变化

React 19 简化了 ref 的传递方式，`forwardRef` 不再是必需的。

### 3.1 ref 作为普通 prop

```jsx
// React 19：ref 可以直接作为 prop 接收
function MyInput({ ref, ...props }) {
  return <input ref={ref} {...props} />;
}

// 使用方式不变
function App() {
  const inputRef = useRef();
  return <MyInput ref={inputRef} />;
}
```

### 3.2 useImperativeHandle 仍然可用

```jsx
// React 19 中 useImperativeHandle 用法不变
function FancyInput({ ref, ...props }) {
  const inputRef = useRef(null);

  useImperativeHandle(ref, () => ({
    focus: () => inputRef.current.focus(),
    clear: () => { inputRef.current.value = ''; },
  }));

  return <input ref={inputRef} {...props} />;
}
```

### 3.3 向后兼容

```jsx
// React 18 中需要 forwardRef
const OldStyle = forwardRef(function OldStyle(props, ref) {
  return <div ref={ref} {...props} />;
});

// React 19 中简写
function NewStyle({ ref, ...props }) {
  return <div ref={ref} {...props} />;
}

// 如果需要同时支持 React 18 和 19，继续使用 forwardRef
```

---

## 四、TypeScript 完整示例

```tsx
import { forwardRef, useImperativeHandle, useRef, useState, ComponentPropsWithoutRef } from 'react';

// 暴露给父组件的 API
export interface ModalHandle {
  open: () => void;
  close: () => void;
  toggle: () => void;
  isOpen: boolean;
}

interface ModalProps extends ComponentPropsWithoutRef<'div'> {
  title: string;
  onClose?: () => void;
  size?: 'small' | 'medium' | 'large';
}

export const Modal = forwardRef<ModalHandle, ModalProps>(function Modal(
  { title, onClose, size = 'medium', children, ...props },
  ref
) {
  const [isOpen, setIsOpen] = useState(false);

  const open = () => setIsOpen(true);
  const close = () => {
    setIsOpen(false);
    onClose?.();
  };
  const toggle = () => setIsOpen(prev => !prev);

  useImperativeHandle(ref, () => ({
    open,
    close,
    toggle,
    get isOpen() { return isOpen; },
  }));

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={close}>
      <div
        className={`modal modal-${size}`}
        onClick={e => e.stopPropagation()}
        {...props}
      >
        <header>
          <h2>{title}</h2>
          <button onClick={close}>×</button>
        </header>
        <div className="modal-body">{children}</div>
      </div>
    </div>
  );
});

// 使用
function App() {
  const modalRef = useRef<ModalHandle>(null);

  return (
    <div>
      <button onClick={() => modalRef.current?.open()}>
        打开弹窗
      </button>

      <Modal ref={modalRef} title="确认" size="medium">
        <p>确定要删除吗？</p>
        <button onClick={() => modalRef.current?.close()}>取消</button>
        <button onClick={() => {
          // 执行删除...
          modalRef.current?.close();
        }}>确认</button>
      </Modal>
    </div>
  );
}
```

---

## 五、forwardRef + memo 组合

```jsx
import { forwardRef, memo } from 'react';

// 同时使用 memo 和 forwardRef
const ExpensiveInput = memo(
  forwardRef(function ExpensiveInput(props, ref) {
    console.log('ExpensiveInput 渲染');
    return <input ref={ref} {...props} />;
  })
);

// props 不变时不会重渲染，同时 ref 仍然可用
function App() {
  const ref = useRef();
  const [value, setValue] = useState('');

  return (
    <div>
      <input value={value} onChange={e => setValue(e.target.value)} />
      <ExpensiveInput ref={ref} placeholder="不会因父组件重渲染而重渲染" />
    </div>
  );
}
```

---

## 六、何时使用 ref 暴露方法

**使用 ref 暴露方法的场景**：
- 聚焦、滚动、选中文本等 DOM 操作
- 表单验证
- 媒体播放控制（play, pause, seek）
- 动画触发
- 第三方库集成

**不应该使用 ref 的场景**：
- 传递数据给子组件：用 props
- 子组件通知父组件：用回调 props
- 共享状态：用 Context 或状态管理
