# 命名空间 namespace

## 一、概念说明

命名空间（`namespace`）是 TypeScript 早期的代码组织方式，用于将相关代码分组到全局命名空间下。**现代项目应优先使用 ES Module**，命名空间只在特定场景（如声明文件）中使用。

## 二、具体用法

### 2.1 基本命名空间

```typescript
namespace Geometry {
  export interface Point {
    x: number;
    y: number;
  }

  export function distance(a: Point, b: Point): number {
    return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
  }

  // 未 export 的成员对外不可见
  function internal(): void {}
}

const p1: Geometry.Point = { x: 0, y: 0 };
const p2: Geometry.Point = { x: 3, y: 4 };
console.log(Geometry.distance(p1, p2));
// 输出: 5
```

### 2.2 命名空间合并

```typescript
namespace Validation {
  export function isString(v: unknown): v is string {
    return typeof v === "string";
  }
}

// 同名命名空间会合并
namespace Validation {
  export function isNumber(v: unknown): v is number {
    return typeof v === "number";
  }
}

console.log(Validation.isString("hello")); // 输出: true
console.log(Validation.isNumber(42));      // 输出: true
```

## 三、注意事项与常见陷阱

1. **优先使用 ES Module**：命名空间是旧模式
2. **必须 `export`**：不导出的成员对外不可见
3. **声明合并**：同名命名空间会合并
4. **仅用于 `.d.ts` 文件**：声明文件中偶尔使用
