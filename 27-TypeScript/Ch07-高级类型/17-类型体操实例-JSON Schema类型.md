# 类型体操实例 - JSON Schema 类型

## 一、概念说明

将 JSON Schema 定义映射为 TypeScript 类型，实现从 Schema 到类型的自动推导。这是一种高级类型编程技巧。

## 二、具体用法

### 2.1 Schema 定义

```typescript
type Schema =
  | { type: "string" }
  | { type: "number" }
  | { type: "boolean" }
  | { type: "null" }
  | { type: "array"; items: Schema }
  | { type: "object"; properties: Record<string, Schema>; required?: string[] };
```

### 2.2 Schema 到类型

```typescript
type FromSchema<S extends Schema> =
  S extends { type: "string" } ? string :
  S extends { type: "number" } ? number :
  S extends { type: "boolean" } ? boolean :
  S extends { type: "null" } ? null :
  S extends { type: "array"; items: infer I extends Schema } ? FromSchema<I>[] :
  S extends { type: "object"; properties: infer P extends Record<string, Schema> }
    ? { [K in keyof P]: FromSchema<P[K]> }
    : never;
```

### 2.3 使用

```typescript
const userSchema = {
  type: "object",
  properties: {
    name: { type: "string" },
    age: { type: "number" },
    active: { type: "boolean" },
  },
} as const satisfies Schema;

type User = FromSchema<typeof userSchema>;
// { name: string; age: number; active: boolean; }

const user: User = { name: "Alice", age: 25, active: true };
console.log(user);
// 输出: { name: "Alice", age: 25, active: true }
```

## 三、注意事项与常见陷阱

1. **`as const` 必须**：保留字面量类型
2. **`satisfies` 验证**：确保 Schema 格式正确
3. **复杂 Schema 性能**：大型 Schema 可能影响编译
4. **可选属性**：需额外处理 `required` 数组
