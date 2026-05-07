# 组件Props类型

## 一、概念说明

Props 是 React 组件的输入接口，TypeScript 通过 `interface` 或 `type` 为 Props 定义类型约束。正确定义 Props 类型是 React + TS 最基础也最重要的技能。

**核心原则**：Props 类型应该描述"这个组件接受什么数据"，清晰且完整。

## 二、具体用法

### 2.1 基本 Props 定义

```tsx
// 推荐使用 interface 定义 Props（可被 extends）
interface ButtonProps {
  variant: 'primary' | 'secondary' | 'danger'; // 字面量联合类型
  size?: 'sm' | 'md' | 'lg';                   // 可选属性
  disabled?: boolean;
  onClick: () => void;                           // 必须的回调
  children: React.ReactNode;                     // 子元素
}

function Button({ variant, size = 'md', disabled, onClick, children }: ButtonProps) {
  return (
    <button
      className={`btn btn-${variant} btn-${size}`}
      disabled={disabled}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

// 使用
<Button variant="primary" onClick={() => console.log('clicked')}>
  点击我
</Button>;
// 编译错误：variant 只能是 'primary' | 'secondary' | 'danger'
// <Button variant="default" onClick={() => {}}>Test</Button>
```

### 2.2 Props 继承与交叉类型

```tsx
// 方式一：interface extends
interface BaseProps {
  className?: string;
  style?: React.CSSProperties;
}

interface CardProps extends BaseProps {
  title: string;
  content: string;
  imageUrl?: string;
}

// 方式二：type 交叉类型
type DivProps = React.HTMLAttributes<HTMLDivElement>;

type PanelProps = DivProps & {
  header: string;
  collapsible?: boolean;
};

function Panel({ header, collapsible, ...divProps }: PanelProps) {
  return <div {...divProps}>{header}</div>;
}
```

### 2.3 children 类型

```tsx
// children 的多种类型选择
type LayoutProps = {
  // ReactNode — 最通用，接受任何可渲染内容
  children: React.ReactNode;

  // ReactElement — 只接受 JSX 元素
  // slot: React.ReactElement;

  // string — 只接受字符串
  // label: string;

  // 函数 children（render props）
  // render: (item: Item) => React.ReactNode;
};

function Layout({ children }: LayoutProps) {
  return <div className="layout">{children}</div>;
}
```

### 2.4 原生 HTML 属性透传

```tsx
// 使用 React.ComponentProps 透传原生属性
type InputProps = React.ComponentProps<'input'> & {
  label: string;
  error?: string;
};

function Input({ label, error, ...inputProps }: InputProps) {
  return (
    <div>
      <label>{label}</label>
      <input {...inputProps} />
      {error && <span className="error">{error}</span>}
    </div>
  );
}

// 可以透传所有原生 input 属性
<Input label="用户名" type="text" placeholder="请输入" maxLength={20} />
```

## 三、注意事项与常见陷阱

1. **优先用 `interface` 定义 Props**：可被扩展，语义更清晰
2. **`children` 在 React 18 后不再是隐式属性**：必须显式声明类型
3. **不要用 `any` 作为 Props 类型**：失去了 TypeScript 的全部意义
4. **利用联合类型限制 Props 取值**：如 `'primary' | 'secondary'` 比 `string` 更安全
5. **原生属性透传用 `ComponentProps`**：避免手动重新声明所有 HTML 属性
