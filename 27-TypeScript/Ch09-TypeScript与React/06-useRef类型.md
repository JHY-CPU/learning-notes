# useRef类型

## 一、概念说明

`useRef` 在 TypeScript 中有两种主要用途：**DOM 引用**和**可变值存储**。两者的类型定义方式不同，理解其区别是正确使用的关键。

`useRef` 返回一个 `MutableRefObject<T>` 对象，其 `.current` 属性是可变的且不会触发重新渲染。

## 二、具体用法

### 2.1 DOM 元素引用

```tsx
import { useRef, useEffect } from 'react';

function InputFocus() {
  // 初始值为 null，泛型指定目标元素类型
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // 必须做 null 检查
    if (inputRef.current) {
      inputRef.current.focus();
    }

    // 或使用可选链
    inputRef.current?.focus();
  }, []);

  return <input ref={inputRef} type="text" />;
}

// 不同元素类型的 Ref
const divRef = useRef<HTMLDivElement>(null);
const canvasRef = useRef<HTMLCanvasElement>(null);
const videoRef = useRef<HTMLVideoElement>(null);
const formRef = useRef<HTMLFormElement>(null);
```

### 2.2 可变值存储（非 DOM）

```tsx
function Timer() {
  // 存储定时器 ID，不需要 DOM 引用
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const startTimer = () => {
    timerRef.current = setTimeout(() => {
      console.log('定时器触发');
    }, 1000);
  };

  const cancelTimer = () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  };

  return (
    <div>
      <button onClick={startTimer}>开始</button>
      <button onClick={cancelTimer}>取消</button>
    </div>
  );
}
```

### 2.3 存储前一次的值

```tsx
function usePrevious<T>(value: T): T | undefined {
  const ref = useRef<T>();

  useEffect(() => {
    ref.current = value;
  }, [value]);

  return ref.current;
}

// 使用
function Counter() {
  const [count, setCount] = useState(0);
  const prevCount = usePrevious(count);

  return (
    <div>
      <p>当前: {count}, 上一次: {prevCount}</p>
      <button onClick={() => setCount(c => c + 1)}>+1</button>
    </div>
  );
}
```

### 2.4 Ref 类型的区别

```tsx
// MutableRefObject — useRef 返回的类型
// .current 是可读写的
const mutRef = useRef<number>(0);
mutRef.current = 42; // OK

// RefObject — 通过 createRef 或回调 ref
// .current 是只读的（React 内部使用）
// 不要在组件内使用 createRef，用 useRef

// RefCallback — 回调 ref
const callbackRef = (node: HTMLInputElement | null) => {
  if (node) {
    node.focus();
  }
};
```

## 三、注意事项与常见陷阱

1. **DOM Ref 初始值为 `null`**：必须用 `useRef<HTMLInputElement>(null)` 而非 `useRef<HTMLInputElement>()`
2. **访问 `.current` 前做 null 检查**：在组件挂载前 `.current` 是 `null`
3. **Ref 变化不触发重新渲染**：修改 `.current` 不会导致组件重新渲染
4. **不要在渲染期间读写 Ref 的值**：只在事件处理或 `useEffect` 中操作
5. **泛型参数不需要 `| null`**：`useRef<HTMLInputElement>(null)` 已经包含了 null 的初始值类型
