# 重新导出 export from

## 一、概念说明

`export { X } from "./module"` 语法将导入的成员**直接重新导出**，不需要先导入再导出。常用于创建 **barrel 文件**（汇总导出文件，通常命名为 `index.ts`），统一模块的对外接口。它简化了模块的组织结构，使消费方可以从一个入口导入多个子模块的成员。

## 二、具体用法

### 2.1 基本重新导出

```typescript
// utils/math.ts
export function add(a: number, b: number): number { return a + b; }
export function multiply(a: number, b: number): number { return a * b; }

// utils/string.ts
export function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}
export function slugify(s: string): string {
  return s.toLowerCase().replace(/\s+/g, "-");
}

// utils/index.ts（barrel 文件）
export { add, multiply } from "./math.js";
export { capitalize, slugify } from "./string.js";

// main.ts — 从 barrel 文件统一导入
import { add, capitalize } from "./utils/index.js";

console.log(add(1, 2));           // 3
console.log(capitalize("hello")); // Hello
```

### 2.2 重命名重新导出

```typescript
export { add as sum } from "./math.js";
export { multiply as product } from "./math.js";
export { capitalize as toUpperFirst } from "./string.js";

// 导入方使用新名称
import { sum, product } from "./utils/index.js";
```

### 2.3 全部重新导出

```typescript
// 重新导出模块的所有命名导出
export * from "./math.js";
export * from "./string.js";
export * from "./types.js";

// 注意：export * 不导出默认导出
// 需要显式导出默认导出
export { default as MathDefault } from "./math.js";
```

### 2.4 类型重新导出

```typescript
// 只重新导出类型
export type { User, Config } from "./types.js";

// 混合值和类型重新导出
export { UserService, API_URL, type User, type Config } from "./services/index.js";
```

### 2.5 嵌套 Barrel 文件

```typescript
// src/components/index.ts
export * from "./Button/index.js";
export * from "./Modal/index.js";
export * from "./Form/index.js";

// src/index.ts（顶层 barrel）
export * from "./components/index.js";
export * from "./utils/index.js";
export * from "./types/index.js";

// 使用方
import { Button, Modal, formatDate } from "my-lib";
```

### 2.6 与 JavaScript 的对比

```javascript
// JavaScript：先导入再导出（两步）
import { add } from "./math.js";
export { add };

// TypeScript/ESM：export from（一步）
export { add } from "./math.js";

// 功能相同，但 export from 更简洁
// 不会在当前模块创建变量绑定
```

## 三、注意事项与常见陷阱

1. **`export *` 不导出默认导出**：需显式 `export { default } from "./mod"` 或 `export { default as X } from "./mod"`
2. **Barrel 文件影响 tree-shaking**：可能导致不必要的代码被包含在 bundle 中
3. **循环依赖**：barrel 文件可能引入 A -> barrel -> A 的循环依赖
4. **大型项目谨慎使用**：对于大型项目，逐模块导入（`from "./utils/math.js"`）可能比 barrel 更高效
5. **`export from` 不创建本地变量**：当前模块中无法使用重新导出的值
6. **命名冲突**：多个模块导出同名成员时，`export *` 会导致冲突报错
