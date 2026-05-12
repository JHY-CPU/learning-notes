# 命名空间 namespace

## 一、概念说明

命名空间（`namespace`）是 TypeScript 早期的代码组织方式，将相关代码分组到全局命名空间下。**现代项目应优先使用 ES Module**，命名空间只在特定场景中使用，如全局类型声明、`.d.ts` 文件中的库声明等。命名空间支持声明合并（同名命名空间自动合并成员）。

## 二、具体用法

### 2.1 基本命名空间

```typescript
namespace Geometry {
  // 必须 export 才对外可见
  export interface Point {
    x: number;
    y: number;
  }

  export function distance(a: Point, b: Point): number {
    return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
  }

  // 未 export 的成员对外不可见（私有）
  function validate(p: Point): boolean {
    return typeof p.x === "number" && typeof p.y === "number";
  }
}

const p1: Geometry.Point = { x: 0, y: 0 };
const p2: Geometry.Point = { x: 3, y: 4 };
console.log(Geometry.distance(p1, p2)); // 5
// Geometry.validate(p1); // 错误：validate 未导出
```

### 2.2 命名空间合并

```typescript
// 同名命名空间自动合并
namespace Validation {
  export function isString(v: unknown): v is string {
    return typeof v === "string";
  }
}

namespace Validation {
  export function isNumber(v: unknown): v is number {
    return typeof v === "number";
  }
}

// 合并后两个函数都可用
console.log(Validation.isString("hello")); // true
console.log(Validation.isNumber(42));      // true
```

### 2.3 嵌套命名空间

```typescript
namespace App {
  export namespace Utils {
    export function log(msg: string): void {
      console.log(`[App] ${msg}`);
    }
  }

  export namespace Models {
    export interface User {
      id: number;
      name: string;
    }
  }
}

App.Utils.log("启动"); // [App] 启动
const user: App.Models.User = { id: 1, name: "Alice" };
```

### 2.4 在 .d.ts 文件中使用

```typescript
// 类型声明文件中常见用法
declare namespace NodeJS {
  interface ProcessEnv {
    NODE_ENV: "development" | "production" | "test";
    PORT: string;
  }
}

// jQuery 风格的库声明
declare namespace JQuery {
  interface AjaxSettings {
    url: string;
    method?: string;
  }
}
```

### 2.5 与 ES Module 的对比

```typescript
// ❌ 命名空间方式（旧）
namespace MathUtils {
  export function add(a: number, b: number) { return a + b; }
  export function multiply(a: number, b: number) { return a * b; }
}

// ✅ ES Module 方式（推荐）
export function add(a: number, b: number): number { return a + b; }
export function multiply(a: number, b: number): number { return a * b; }

// ES Module 优势：
// 1. 标准化的 JavaScript 特性
// 2. 支持 tree-shaking
// 3. 有明确的依赖关系
// 4. 支持按需加载
```

### 2.6 与 JavaScript 的对比

```javascript
// JavaScript 没有 namespace，用对象模拟
const Geometry = {
  distance(a, b) {
    return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
  },
};

// TypeScript namespace 编译后也是类似的对象
// 但 namespace 更适合类型声明场景
```

## 三、注意事项与常见陷阱

1. **优先使用 ES Module**：namespace 是旧模式，现代项目不应使用 namespace 组织代码
2. **必须 `export`**：不导出的成员对外不可见，容易忘记
3. **声明合并**：同名命名空间会自动合并，可能导致意外的成员混入
4. **仅用于 `.d.ts` 文件**：声明文件中声明全局类型空间时偶尔使用
5. **不能导出类型**：namespace 中不能使用 `export type`，接口等类型默认可导出
6. **与模块互斥**：文件中有 `import`/`export` 时，namespace 变为模块级而非全局
