# typeof 运算符

## 一、概念说明

TypeScript 中 `typeof` 有**双重用途**：在值空间中（运行时）它是 JavaScript 的 `typeof` 运算符，返回值的类型字符串；在类型空间中（编译时）它从已有的值或变量中提取 TypeScript 类型。后者是类型编程的常用工具，配合 `ReturnType`、`Parameters` 等工具类型，可以实现从值到类型的自动推导。

## 二、具体用法

### 2.1 类型空间的 typeof（从值提取类型）

```typescript
const config = {
  host: "localhost",
  port: 3000,
  debug: true,
  tags: ["dev", "local"],
};

// 从值获取其精确类型
type Config = typeof config;
// { host: string; port: number; debug: boolean; tags: string[] }

// 注意：let 声明会拓宽类型
let x = "hello";      // x: string（不是 "hello"）
const y = "hello";     // y: "hello"（字面量类型）
type TX = typeof x;    // string
type TY = typeof y;    // "hello"
```

### 2.2 配合 ReturnType 和 Parameters

```typescript
function createUser(name: string, age: number) {
  return { id: Math.random(), name, age, createdAt: new Date() };
}

// 提取返回类型（不需要手动定义接口）
type User = ReturnType<typeof createUser>;
// { id: number; name: string; age: number; createdAt: Date }

// 提取参数类型
type CreateUserParams = Parameters<typeof createUser>;
// [string, number]

// 实际应用：类型安全的 API 响应
async function fetchUsers() {
  const res = await fetch("/api/users");
  return res.json() as Promise<{ users: User[] }>;
}
type FetchUsersReturn = ReturnType<typeof fetchUsers>;
// Promise<{ users: User[] }>
```

### 2.3 typeof 从导入模块提取类型

```typescript
// 从第三方库的值提取类型
import { EventEmitter } from "events";
type EmitterEvents = InstanceType<typeof EventEmitter>;

// 从配置对象提取
import defaultConfig from "./config.json";
type AppConfig = typeof defaultConfig;
// 确保自定义配置与默认配置结构一致
function mergeConfig(custom: Partial<AppConfig>): AppConfig {
  return { ...defaultConfig, ...custom };
}
```

### 2.4 typeof 在 switch 中（运行时类型守卫）

```typescript
function describe(value: string | number | boolean | object): string {
  switch (typeof value) {
    case "string":  return `字符串: "${value}"（长度 ${value.length}）`;
    case "number":  return `数字: ${value}`;
    case "boolean": return `布尔: ${value}`;
    case "object":  return `对象: ${JSON.stringify(value)}`;
    default:        return `未知类型: ${typeof value}`;
  }
}

console.log(describe("hello"));  // 字符串: "hello"（长度 5）
console.log(describe(42));       // 数字: 42
console.log(describe({ a: 1 })); // 对象: {"a":1}
```

### 2.5 深层 typeof

```typescript
const routes = {
  home: { path: "/", title: "首页" },
  user: { path: "/user/:id", title: "用户" },
  admin: { path: "/admin", title: "管理" },
};

// 提取嵌套类型
type Routes = typeof routes;
type HomeRoute = Routes["home"];       // { path: "/"; title: "首页" }
type HomePath = Routes["home"]["path"]; // "/"

// 提取所有路径联合
type AllPaths = Routes[keyof Routes]["path"];
// "/" | "/user/:id" | "/admin"
```

### 2.6 与 JavaScript 的对比

```javascript
// JavaScript typeof：运行时检查，返回字符串
typeof "hello"   // "string"
typeof 42        // "number"
typeof null      // "object"（历史遗留 bug）
typeof undefined // "undefined"

// TypeScript typeof：编译时提取类型
// const obj = { a: 1, b: "hello" };
// type ObjType = typeof obj; // { a: number; b: string }
// 这是完全不同的操作——提取的是类型信息
```

## 三、注意事项与常见陷阱

1. **类型空间 vs 值空间**：`typeof` 在注解位置是类型操作，在表达式中是运行时操作
2. **`typeof` 只能用于值**：不能对类型或接口使用 `typeof`，如 `typeof string` 是非法的
3. **`let` 拓宽类型**：`let x = "hello"` 的 `typeof x` 是 `string` 而非 `"hello"`，用 `as const` 或 `const` 保留字面量
4. **`typeof null === "object"`**：JavaScript 历史遗留问题，类型守卫中需额外检查
5. **结合工具类型**：`ReturnType<typeof fn>`、`Parameters<typeof fn>` 是最常见用法
6. **不能用于泛型函数**：`typeof identity` 会丢失泛型信息，返回 `(...args: any[]) => any`
