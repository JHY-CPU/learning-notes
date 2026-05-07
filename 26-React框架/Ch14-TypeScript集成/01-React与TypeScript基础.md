# React 与 TypeScript 基础

## 1. 为什么在 React 中使用 TypeScript

TypeScript 为 React 项目带来了编译时类型安全，能够在编码阶段捕获大量常见错误。

### 核心优势

| 优势 | 说明 |
|------|------|
| **类型安全** | Props、State、事件对象等在编译时检查 |
| **IDE 支持** | 自动补全、重构、跳转定义 |
| **文档化** | 类型即文档，Props 接口即组件 API 文档 |
| **重构友好** | 修改接口后编译器会指出所有需要更新的地方 |
| **团队协作** | 减少因类型不匹配导致的 Bug |

```tsx
// 没有 TypeScript：运行时才发现错误
function Greeting({ name, age }) {
  return <div>{name.toUpperCase()} is {age.toFixed(1)} years old</div>
}
// 如果传入 age 为字符串，运行时才报错

// 有 TypeScript：编译时就报错
interface GreetingProps {
  name: string
  age: number
}
function Greeting({ name, age }: GreetingProps) {
  return <div>{name.toUpperCase()} is {age.toFixed(1)} years old</div>
}
// <Greeting name={123} /> 编译报错
```

---

## 2. tsconfig.json 配置（React 项目）

React 项目推荐使用 Vite 创建 TypeScript 项目：

```bash
npm create vite@latest my-app -- --template react-ts
```

### 关键 tsconfig 选项

```jsonc
{
  "compilerOptions": {
    // 基础配置
    "target": "ES2020",                    // 编译目标
    "lib": ["ES2020", "DOM", "DOM.Iterable"], // 包含的类型定义
    "module": "ESNext",                     // 模块系统
    "moduleResolution": "bundler",          // 模块解析策略（Vite 推荐）

    // JSX 配置
    "jsx": "react-jsx",                     // React 17+ 的 JSX 转换（无需 import React）
    "jsxImportSource": "@emotion/react",    // 可选：自定义 JSX 运行时

    // 严格模式
    "strict": true,                         // 开启所有严格检查
    "noUnusedLocals": true,                 // 未使用的局部变量报错
    "noUnusedParameters": true,             // 未使用的参数报错
    "noFallthroughCasesInSwitch": true,     // switch 穿透报错

    // 路径别名
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"]
    },

    // 其他
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true                          // 不输出文件（由构建工具处理）
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

### `strict: true` 包含的子选项

```
strict
├── strictNullChecks        // null 和 undefined 需要显式处理
├── strictFunctionTypes     // 函数参数逆变检查
├── strictBindCallApply     // bind/call/apply 严格检查
├── strictPropertyInitialization // 类属性必须初始化
├── noImplicitAny           // 禁止隐式 any
├── noImplicitThis          // 禁止隐式 this
└── alwaysStrict            // 始终使用 "use strict"
```

---

## 3. `.tsx` 文件

### 文件扩展名约定

| 扩展名 | 用途 |
|--------|------|
| `.ts` | 纯 TypeScript 文件（工具函数、类型定义等） |
| `.tsx` | 包含 JSX 的 TypeScript 文件（React 组件） |
| `.d.ts` | 类型声明文件 |

**规则**：只要文件中包含 JSX 语法，就必须使用 `.tsx` 扩展名。

```tsx
// src/components/Button.tsx
export function Button({ label }: { label: string }) {
  return <button>{label}</button>
}
```

```ts
// src/utils/format.ts — 纯函数，用 .ts
export function formatDate(date: Date): string {
  return date.toLocaleDateString()
}
```

---

## 4. 类型化 JSX

### JSX 的类型本质

```tsx
// JSX 最终被编译为 React.createElement 调用
const element = <h1>Hello</h1>
// 编译后 ↓
const element = React.createElement("h1", null, "Hello")
// 或（react-jsx 模式）↓
import { jsx } from "react/jsx-runtime"
const element = jsx("h1", { children: "Hello" })
```

### 常用 JSX 类型

```tsx
// JSX.Element — 具体的 JSX 元素类型
const element: JSX.Element = <div>Hello</div>

// JSX.IntrinsicElements — 内置 HTML 元素的类型映射
// 内部大致如下（简化版）：
namespace JSX {
  interface IntrinsicElements {
    div: React.HTMLAttributes<HTMLDivElement>
    button: React.ButtonHTMLAttributes<HTMLButtonElement>
    input: React.InputHTMLAttributes<HTMLInputElement>
    // ... 所有 HTML/SVG 元素
  }
}
```

---

## 5. `React.FC` 类型

### 定义与用法

`React.FC`（Function Component）是 React 内置的泛型类型，用于标注函数组件的类型。

```tsx
// 使用 React.FC
interface Props {
  name: string
  age?: number
}

const Greeting: React.FC<Props> = ({ name, age }) => {
  return <div>{name} {age}</div>
}

// 等价于
const Greeting = ({ name, age }: Props): React.JSX.Element => {
  return <div>{name} {age}</div>
}
```

### `React.FC` 的特点

```tsx
// 1. 自动包含 children prop（React 18 之前）
interface Props {
  title: string
}
const Card: React.FC<Props> = ({ title, children }) => {
  // React 18 之前：children 自动可用
  return <div><h2>{title}</h2>{children}</div>
}

// 2. React 18+：children 不再自动包含
// 需要显式声明
interface Props {
  title: string
  children?: React.ReactNode  // 必须手动添加
}
const Card: React.FC<Props> = ({ title, children }) => {
  return <div><h2>{title}</h2>{children}</div>
}
```

### 是否推荐使用 `React.FC`？

**社区争议**：许多开发者倾向于不使用 `React.FC`，原因如下：

```tsx
// ❌ 不使用 React.FC（更简洁，更灵活）
interface ButtonProps {
  label: string
  onClick: () => void
  variant?: 'primary' | 'secondary'
}

function Button({ label, onClick, variant = 'primary' }: ButtonProps) {
  return (
    <button
      className={`btn btn-${variant}`}
      onClick={onClick}
    >
      {label}
    </button>
  )
}

// ✅ 使用 React.FC（显式返回类型）
const Button: React.FC<ButtonProps> = ({ label, onClick, variant = 'primary' }) => {
  return (
    <button
      className={`btn btn-${variant}`}
      onClick={onClick}
    >
      {label}
    </button>
  )
}
```

**建议**：初学者可以用 `React.FC`，熟练后推荐直接标注 Props 类型，不包裹 FC。

---

## 6. `React.ReactNode` vs `JSX.Element` vs `React.ReactElement`

这三个类型经常混淆，它们的区别非常重要：

```tsx
// React.ReactNode — 最宽泛，推荐用于 children
// 包含：ReactElement | string | number | boolean | null | undefined | ReactFragment | ReactPortal
type ReactNode =
  | ReactElement
  | string
  | number
  | boolean
  | null
  | undefined
  | Iterable<ReactNode>
  | ReactPortal

// JSX.Element — 等同于 ReactElement<any, any>
// 表示一个具体的 JSX 元素
const element: JSX.Element = <div>Hello</div>

// React.ReactElement<P, T> — 最具体的类型
// P = props 类型, T = 元素类型
const element: ReactElement<{ id: string }, 'div'> = <div id="1">Hello</div>
```

### 使用场景对比

```tsx
interface CardProps {
  // ✅ 用 React.ReactNode — 接受最广泛的内容类型
  children: React.ReactNode

  // ✅ 用 React.ReactElement — 只接受单个 React 元素
  icon: React.ReactElement

  // ❌ 用 JSX.Element — 同 ReactElement，但不携带 props 类型信息
  header: JSX.Element
}

// 实际使用
function Card({ children, icon, header }: CardProps) {
  return (
    <div>
      {icon}           {/* 必须是 <Icon /> 这样的元素 */}
      {header}         {/* 必须是 React 元素 */}
      {children}       {/* 可以是任何内容：文字、数字、元素、数组等 */}
    </div>
  )
}

<Card
  icon={<StarIcon />}
  header={<h2>Title</h2>}
>
  <p>这是一段文字</p>           {/* ReactNode ✅ */}
  {/* "纯文字" 也是 ReactNode ✅ */}
  {/* 42 也是 ReactNode ✅ */}
  {/* null 也是 ReactNode ✅ */}
</Card>
```

### 推荐使用总结

| 场景 | 推荐类型 |
|------|----------|
| `children` prop | `React.ReactNode` |
| 要求传入单个元素 | `React.ReactElement` |
| 组件返回值 | `React.ReactNode` 或自动推断 |
| 需要精确 props 类型 | `React.ReactElement<Props, T>` |

---

## 7. `type` vs `interface` 定义 Props

### 基本对比

```tsx
// interface 方式
interface ButtonProps {
  label: string
  onClick: () => void
  disabled?: boolean
}

// type 方式
type ButtonProps = {
  label: string
  onClick: () => void
  disabled?: boolean
}
```

### 关键差异

```tsx
// 1. interface 可以声明合并（同名 interface 会合并）
interface UserInfo {
  name: string
}
interface UserInfo {
  age: number
}
// 合并后：{ name: string; age: number }

// type 不能重复声明
type UserInfo = { name: string }
// type UserInfo = { age: number } // ❌ Error: Duplicate identifier

// 2. interface 支持 extends
interface BaseProps {
  className?: string
}
interface ButtonProps extends BaseProps {
  label: string
}

// 3. type 支持联合类型、交叉类型等
type BaseProps = {
  className?: string
}
type ButtonProps = BaseProps & {
  label: string
}

// type 支持联合类型（interface 不行）
type Status = 'loading' | 'success' | 'error'
type Props = LoadingProps | SuccessProps | ErrorProps

// 4. type 支持映射类型、条件类型等高级特性
type ReadonlyProps<T> = {
  readonly [K in keyof T]: T[K]
}
```

### React Props 场景推荐

```tsx
// ✅ 简单 Props：type 和 interface 都可以，选择你团队的约定
type SimpleProps = {
  title: string
  count: number
}

// ✅ 需要联合类型（变体组件）：必须用 type
type AlertProps = {
  variant: 'success'
  message: string
} | {
  variant: 'error'
  message: string
  errorCode: number
}

// ✅ 需要交叉类型：用 type 或 interface extends
type EnhancedButton = BaseButtonProps & {
  loading?: boolean
}

// ✅ 泛型 Props：两者都行
interface ListProps<T> {
  items: T[]
  renderItem: (item: T) => React.ReactNode
}
```

**团队建议**：统一使用一种。许多团队选择 Props 用 `type`，其他场景用 `interface`。

---

## 8. 项目搭建最佳实践

### 目录结构

```
src/
├── types/              # 全局类型定义
│   ├── index.ts        # 导出所有类型
│   ├── api.ts          # API 相关类型
│   └── components.ts   # 公共组件 Props 类型
├── components/         # 组件
│   ├── Button/
│   │   ├── Button.tsx
│   │   ├── Button.test.tsx
│   │   ├── Button.module.css
│   │   └── index.ts
├── hooks/              # 自定义 Hooks
├── utils/              # 工具函数
└── App.tsx
```

### 常见类型定义文件

```ts
// src/types/index.ts

// 全局共享类型
export interface User {
  id: string
  name: string
  email: string
  avatar?: string
}

export interface ApiResponse<T> {
  data: T
  status: number
  message: string
}

export type Nullable<T> = T | null
export type Optional<T> = T | undefined
export type AsyncState<T> = {
  data: T | null
  loading: boolean
  error: Error | null
}
```

### CSS Modules 类型声明

```ts
// src/vite-env.d.ts
/// <reference types="vite/client" />

// CSS Modules 类型声明
declare module '*.module.css' {
  const classes: { readonly [key: string]: string }
  export default classes
}

declare module '*.module.scss' {
  const classes: { readonly [key: string]: string }
  export default classes
}

// 静态资源类型声明
declare module '*.svg' {
  import React from 'react'
  export const ReactComponent: React.FC<React.SVGProps<SVGSVGElement>>
  const src: string
  export default src
}

declare module '*.png' {
  const src: string
  export default src
}

declare module '*.jpg' {
  const src: string
  export default src
}
```

---

## 9. 常见错误与解决

```tsx
// ❌ 错误：Type 'undefined' is not assignable to type 'ReactNode'
const App = () => {
  const result = someFunction() // 返回类型可能是 undefined
  return <div>{result}</div>     // TS 报错
}

// ✅ 解决：使用条件渲染
const App = () => {
  const result = someFunction()
  return <div>{result ?? ''}</div>
}

// ❌ 错误：Property 'children' does not exist
const Wrapper: React.FC<{ title: string }> = ({ title, children }) => {
  // React 18+ FC 不自动包含 children
  return <div><h1>{title}</h1>{children}</div>
}

// ✅ 解决：显式声明 children
interface WrapperProps {
  title: string
  children?: React.ReactNode
}
const Wrapper: React.FC<WrapperProps> = ({ title, children }) => {
  return <div><h1>{title}</h1>{children}</div>
}

// ❌ 错误：不能将 HTML 属性传递给组件
<div onClick={() => {}} className="btn" />

// ✅ 解决：组件接收 onClick 需要声明类型
interface ButtonProps {
  onClick?: React.MouseEventHandler<HTMLButtonElement>
}
```

---

## 总结

- **tsconfig.json** 中 `strict: true` 是必须的
- `.tsx` 文件用于包含 JSX 的代码
- **children prop** 类型使用 `React.ReactNode`
- Props 定义 `type` 和 `interface` 都可以，团队统一即可
- 养成写类型的习惯，从项目一开始就在 TypeScript 中开发
