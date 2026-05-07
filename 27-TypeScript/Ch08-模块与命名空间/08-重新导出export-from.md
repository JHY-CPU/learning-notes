# 重新导出 export from

## 一、概念说明

`export { X } from "./module"` 语法将导入的成员直接重新导出，不需要先导入再导出。常用于创建 barrel 文件（汇总导出）。

## 二、具体用法

### 2.1 基本重新导出

```typescript
// utils/math.ts
export function add(a: number, b: number): number {
  return a + b;
}

// utils/string.ts
export function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

// utils/index.ts（barrel 文件）
export { add } from "./math.js";
export { capitalize } from "./string.js";

// main.ts - 从 barrel 文件统一导入
import { add, capitalize } from "./utils/index.js";

console.log(add(1, 2));           // 输出: 3
console.log(capitalize("hello")); // 输出: Hello
```

### 2.2 重命名重新导出

```typescript
export { add as sum } from "./math.js";
export { capitalize as toUpperFirst } from "./string.js";
```

### 2.3 全部重新导出

```typescript
export * from "./math.js";
export * from "./string.js";
```

## 三、注意事项与常见陷阱

1. **`export *` 不导出默认导出**：需要显式 `export { default } from`
2. **Barrel 文件影响 tree-shaking**：可能导致不必要的代码被包含
3. **循环依赖**：barrel 文件可能引入循环依赖
4. **大型项目谨慎使用**：逐模块导入可能更优
