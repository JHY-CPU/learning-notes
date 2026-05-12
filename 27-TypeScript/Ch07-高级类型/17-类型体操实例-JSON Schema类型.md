# 类型体操实例 - JSON Schema 类型

## 一、概念说明

将 JSON Schema 定义映射为 TypeScript 类型，实现从 Schema 到类型的**自动推导**。这是一种高级类型编程技巧，使用条件类型和递归 `infer` 解析 Schema 结构，自动推导出对应的 TypeScript 接口。类似原理应用于 `zod`、`yup` 等运行时验证库的类型推导。

## 二、具体用法

### 2.1 Schema 类型定义

```typescript
// JSON Schema 的 TypeScript 表示
type Schema =
  | { type: "string" }
  | { type: "number" }
  | { type: "boolean" }
  | { type: "null" }
  | { type: "array"; items: Schema }
  | {
      type: "object";
      properties: Record<string, Schema>;
      required?: string[];
    };
```

### 2.2 Schema 到类型的映射

```typescript
type FromSchema<S extends Schema> =
  S extends { type: "string" } ? string :
  S extends { type: "number" } ? number :
  S extends { type: "boolean" } ? boolean :
  S extends { type: "null" } ? null :
  S extends { type: "array"; items: infer I extends Schema }
    ? FromSchema<I>[]
    : S extends { type: "object"; properties: infer P extends Record<string, Schema> }
      ? { [K in keyof P]: FromSchema<P[K]> }
      : never;
```

### 2.3 使用示例

```typescript
// 定义 Schema（使用 as const + satisfies 保留字面量）
const userSchema = {
  type: "object",
  properties: {
    name: { type: "string" },
    age: { type: "number" },
    active: { type: "boolean" },
    tags: { type: "array", items: { type: "string" } },
  },
} as const satisfies Schema;

// 自动推导出 TypeScript 类型
type User = FromSchema<typeof userSchema>;
// {
//   name: string;
//   age: number;
//   active: boolean;
//   tags: string[];
// }

const user: User = {
  name: "Alice",
  age: 25,
  active: true,
  tags: ["admin", "user"],
};
```

### 2.4 嵌套对象 Schema

```typescript
const configSchema = {
  type: "object",
  properties: {
    server: {
      type: "object",
      properties: {
        host: { type: "string" },
        port: { type: "number" },
      },
    },
    debug: { type: "boolean" },
  },
} as const satisfies Schema;

type Config = FromSchema<typeof configSchema>;
// {
//   server: { host: string; port: number };
//   debug: boolean;
// }
```

### 2.5 处理可选属性

```typescript
// 增强版：支持 required 字段
type FromSchemaWithRequired<S extends Schema> =
  S extends { type: "object"; properties: infer P extends Record<string, Schema>; required: infer R extends string[] }
    ? { [K in keyof P as K extends R[number] ? K : never]: FromSchema<P[K]> }
      & { [K in keyof P as K extends R[number] ? never : K]?: FromSchema<P[K]> }
    : FromSchema<S>;

// 简化版：所有属性必选
type SimplifiedSchema = {
  type: "object";
  properties: Record<string, Schema>;
  required?: string[];
};
```

### 2.6 实际应用：类型安全的配置校验

```typescript
// 根据 Schema 运行时验证
function validateBySchema<T extends Schema>(
  data: unknown,
  schema: T
): data is FromSchema<T> {
  if (schema.type === "object" && typeof data === "object" && data !== null) {
    const obj = data as Record<string, unknown>;
    const props = (schema as any).properties as Record<string, Schema>;
    return Object.entries(props).every(([key, propSchema]) =>
      validateBySchema(obj[key], propSchema)
    );
  }
  return typeof data === schema.type;
}

// 使用
const data: unknown = { name: "Alice", age: 25, active: true, tags: ["admin"] };
if (validateBySchema(data, userSchema)) {
  // data 类型收窄为 User
  console.log(data.name); // string，类型安全
}
```

### 2.7 与 JavaScript 的对比

```javascript
// JavaScript：运行时根据 Schema 校验
const Ajv = require("ajv");
const ajv = new Ajv();
const validate = ajv.compile({
  type: "object",
  properties: { name: { type: "string" }, age: { type: "number" } },
});
const valid = validate({ name: "Alice", age: 25 });

// TypeScript FromSchema：编译时根据 Schema 推导类型
// 不依赖运行时校验库，编译时就确定类型
```

## 三、注意事项与常见陷阱

1. **`as const` 必须**：不加 `as const` 字面量会拓宽，`{ type: "string" }` 变为 `{ type: string }` 无法精确匹配
2. **`satisfies` 验证**：确保 Schema 定义符合 `Schema` 类型约束，提前发现格式错误
3. **复杂 Schema 性能**：大型深层嵌套 Schema 可能显著影响编译速度
4. **可选属性需要额外处理**：基础版不处理 `required` 数组，需增强 `FromSchema`
5. **联合类型 Schema**：支持 `oneOf`、`anyOf` 需更复杂的递归逻辑
6. **引用（$ref）**：JSON Schema 的引用机制需要额外实现
