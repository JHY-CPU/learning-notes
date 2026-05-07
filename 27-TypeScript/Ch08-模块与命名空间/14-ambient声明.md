# ambient 声明

## 一、概念说明

Ambient 声明（`declare`）用于告诉 TypeScript 存在但未在当前代码中定义的变量、函数、模块等。常用于声明全局变量、CSS 模块、非 TypeScript 模块的类型。

## 二、具体用法

### 2.1 声明全局变量

```typescript
declare const API_URL: string;
declare function setTimeout(callback: () => void, ms: number): number;

// 不需要导入，直接使用
console.log(API_URL);
```

### 2.2 声明模块

```typescript
// 为 .css 文件声明模块
declare module "*.css" {
  const classes: Record<string, string>;
  export default classes;
}

// 为 .svg 文件声明模块
declare module "*.svg" {
  const content: string;
  export default content;
}
```

### 2.3 声明非 TS 模块

```typescript
// 为没有类型的 JS 库声明类型
declare module "legacy-library" {
  export function doSomething(x: number): string;
  export const VERSION: string;
}
```

### 2.4 声明枚举

```typescript
declare enum LogLevel {
  Debug,
  Info,
  Warn,
  Error,
}
```

## 三、注意事项与常见陷阱

1. **`declare` 不生成代码**：只提供类型信息
2. **必须在 `.d.ts` 文件或模块中**：普通 `.ts` 文件中通常不需要
3. **全局声明的影响**：全局可用，注意命名冲突
4. **`declare module "*"` 通配符**：匹配特定文件扩展名
