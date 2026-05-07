# TypeScript与React概述

## 一、概念说明

TypeScript 为 React 应用提供了静态类型检查能力，能够在编码阶段发现潜在错误。React 本身是用 JavaScript 编写的，但官方提供了完整的 TypeScript 类型定义，使得两者结合使用非常自然。

**为什么要用 TypeScript 写 React：**

1. **Props 类型安全**：组件的输入输出都有明确的类型约束
2. **IDE 智能提示**：自动补全 Props、事件类型、组件属性
3. **重构更安全**：修改类型后编译器会标记所有受影响的代码
4. **自文档化**：类型本身就是组件 API 的文档

## 二、具体用法

### 2.1 基本对比：JS vs TS

```tsx
// JavaScript 版本 — 没有类型检查，容易传错 props
function UserCard({ name, age }) {
  return <div>{name}, {age}岁</div>;
}
// 错误传参不会被发现
<UserCard name={123} age="张三" />

// TypeScript 版本 — 编译时就能发现错误
interface UserCardProps {
  name: string;
  age: number;
}

function UserCard({ name, age }: UserCardProps) {
  return <div>{name}, {age}岁</div>;
}
// 编译报错：类型 "number" 的参数不能赋给类型 "string" 的参数
// <UserCard name={123} age="张三" />
```

### 2.2 常用类型工具

```tsx
// React 内置的类型工具
import React, { FC, ReactNode, ReactElement } from 'react';

// FC — 函数组件类型（已包含 children）
const Greet: FC<{ name: string }> = ({ name }) => <h1>Hello, {name}</h1>;

// ReactNode — 可渲染的所有类型
type SlotProps = {
  content: ReactNode; // string | number | ReactElement | ReactNode[]
};

// ReactElement — 元素类型
type WrapperProps = {
  child: ReactElement; // 只接受 JSX 元素
};
```

### 2.3 项目技术栈选择

| 方案 | 适用场景 | 说明 |
|------|----------|------|
| Vite + React + TS | 新项目首选 | 快速、配置简洁 |
| Next.js + TS | SSR/全栈项目 | 内置路由和 SSR |
| Remix + TS | 全栈应用 | 嵌套路由、数据加载 |
| CRA + TS | 学习入门 | 已不推荐用于生产 |

## 三、注意事项与常见陷阱

1. **不要过度类型化**：React 和 TypeScript 都能自动推断的类型，不需要手动标注
2. **使用 `interface` 定义 Props**：可以被扩展（`extends`），更适合组件 Props
3. **使用 `type` 定义内部类型**：联合类型、工具类型等场景
4. **保持 TypeScript 版本更新**：React 类型定义会使用最新 TS 特性
5. **安装正确的类型包**：React 18+ 自带类型，旧版本需要 `@types/react`
