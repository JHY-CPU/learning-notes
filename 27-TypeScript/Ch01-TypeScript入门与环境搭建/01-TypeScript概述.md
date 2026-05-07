# TypeScript概述

## 一、概念说明

TypeScript 是由微软开发的开源编程语言，它是 JavaScript 的**超集**（Superset），意味着所有合法的 JavaScript 代码都是合法的 TypeScript 代码。TypeScript 在 JavaScript 基础上添加了**静态类型系统**、**类**、**接口**、**泛型**等特性，并通过编译器（`tsc`）将 TypeScript 代码编译为纯 JavaScript 运行。

### 1.1 与 JavaScript 的关系

- **JavaScript** 是动态类型语言，变量类型在运行时确定
- **TypeScript** 在编译阶段进行类型检查，捕获潜在错误
- TypeScript 最终编译为 JavaScript，可在任何 JS 运行环境执行

### 1.2 TypeScript 的核心优势

| 优势 | 说明 |
|------|------|
| 类型安全 | 编译时捕获类型错误，减少运行时bug |
| 智能提示 | IDE 提供精确的自动补全和文档 |
| 代码重构 | 类型信息让重构更安全可靠 |
| 更好的可读性 | 类型注解即文档，提升代码可维护性 |
| 生态兼容 | 完全兼容 JavaScript 生态系统 |

## 二、具体用法

### 2.1 TypeScript vs JavaScript 对比

```typescript
// JavaScript - 运行时才能发现类型错误
function add(a, b) {
  return a + b;
}
add("1", 2); // "12" - 字符串拼接，不易察觉的bug

// TypeScript - 编译时即捕获错误
function addTs(a: number, b: number): number {
  return a + b;
}
// addTs("1", 2); // 编译错误: Argument of type 'string' is not assignable
addTs(1, 2); // 3 - 正确用法
```

**输出：**
```
// 编译阶段 TypeScript 会报错：
// TS2345: Argument of type 'string' is not assignable to parameter of type 'number'.
```

### 2.2 TypeScript 适用场景

```typescript
// 接口定义 - 明确数据结构
interface User {
  id: number;
  name: string;
  email: string;
  age?: number; // 可选属性
}

// 泛型 - 类型安全的容器
function getFirst<T>(arr: T[]): T | undefined {
  return arr[0];
}

const num = getFirst([1, 2, 3]);       // 类型推断为 number
const str = getFirst(["a", "b", "c"]); // 类型推断为 string
```

**输出：**
```
num 的类型为: number | undefined
str 的类型为: string | undefined
```

## 三、注意事项与常见陷阱

1. **TypeScript 不是新语言**：它只是给 JavaScript 加了类型系统，最终产物仍是 JS
2. **编译步骤**：浏览器不能直接运行 `.ts` 文件，必须先编译为 `.js`
3. **类型擦除**：运行时不存在类型信息，`interface` 和 `type` 编译后会消失
4. **渐进采用**：可以用 `any` 逐步迁移现有 JavaScript 项目
5. **学习曲线**：需要额外学习类型系统语法，但收益远大于投入
6. **不是万能的**：TypeScript 只做静态检查，运行时错误仍需手动防范
