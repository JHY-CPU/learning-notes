# infer 推断详解

## 一、概念说明

`infer` 是 TypeScript 条件类型中的**类型变量声明**关键字，只能在 `extends` 子句中使用。它从复杂类型结构中"提取"或"捕获"子类型，类似于正则表达式的捕获组。`infer` 是类型模式匹配的核心工具，`ReturnType`、`Parameters`、`Awaited` 等内置工具类型都基于它实现。

## 二、具体用法

### 2.1 提取函数返回类型与参数类型

```typescript
// 提取返回类型
type MyReturnType<T> = T extends (...args: any[]) => infer R ? R : never;
type R1 = MyReturnType<() => string>;           // string
type R2 = MyReturnType<(x: number) => boolean>; // boolean

// 提取参数类型（元组形式）
type MyParameters<T> = T extends (...args: infer P) => any ? P : never;
type P1 = MyParameters<(x: number, y: string) => void>; // [number, string]

// 实际应用
function createUser(name: string, age: number) {
  return { id: 1, name, age, createdAt: new Date() };
}
type User = MyReturnType<typeof createUser>;
// { id: number; name: string; age: number; createdAt: Date }
type CreateUserArgs = MyParameters<typeof createUser>;
// [string, number]
```

### 2.2 递归解包 Promise

```typescript
type MyAwaited<T> = T extends Promise<infer U> ? MyAwaited<U> : T;

type A1 = MyAwaited<Promise<string>>;                // string
type A2 = MyAwaited<Promise<Promise<number>>>;       // number
type A3 = MyAwaited<Promise<Promise<Promise<boolean>>>>; // boolean

// 实际应用
async function fetchUser() {
  return { id: 1, name: "Alice" };
}
type UserData = MyAwaited<ReturnType<typeof fetchUser>>;
// { id: number; name: string }
```

### 2.3 提取数组/元组元素类型

```typescript
type Element<T> = T extends (infer E)[] ? E : never;
type E1 = Element<string[]>;              // string
type E2 = Element<[number, boolean, string]>; // number | boolean | string

// 提取首尾元素
type Head<T extends any[]> = T extends [infer H, ...any] ? H : never;
type Tail<T extends any[]> = T extends [any, ...infer R] ? R : never;

type H = Head<[1, 2, 3]>; // 1
type T = Tail<[1, 2, 3]>; // [2, 3]
```

### 2.4 提取字符串模式

```typescript
// 提取 CSS 值的数字部分
type ExtractPx<T extends string> = T extends `${infer N}px` ? N : never;
type N1 = ExtractPx<"16px">;  // "16"
type N2 = ExtractPx<"100px">; // "100"

// 提取路径参数
type ExtractParam<T extends string> =
  T extends `/:${infer Param}` ? Param : never;
type Param = ExtractParam<"/userId">; // "userId"

// 反转字符串（递归）
type ReverseString<S extends string> =
  S extends `${infer H}${infer T}` ? `${ReverseString<T>}${H}` : "";
type Rev = ReverseString<"hello">; // "olleh"
```

### 2.5 逆变位置的 infer（交叉类型）

```typescript
// 函数参数位置是逆变的，多个 infer 结果为交叉类型
type UnionToIntersection<T> =
  (T extends any ? (x: T) => void : never) extends
  (x: infer R) => void ? R : never;

type U2I = UnionToIntersection<{ a: 1 } | { b: 2 }>;
// { a: 1 } & { b: 2 }

// 实际应用：合并多个配置类型
type Config = UnionToIntersection<
  | { server: { port: number } }
  | { db: { url: string } }
>;
// { server: { port: number } } & { db: { url: string } }
```

### 2.6 与 JavaScript 的对比

```javascript
// JavaScript：运行时解构提取值
const response = { data: { user: { name: "Alice" } } };
const { data: { user: { name } } } = response;
// name = "Alice"

// TypeScript infer：编译时提取类型
// type Name = Response extends { data: { user: { name: infer N } } } ? N : never;
// => "Alice" (如果 Response 是字面量类型)
// 这是类型层面的"解构"
```

## 三、注意事项与常见陷阱

1. **只能在 `extends` 子句中使用**：不能单独使用 `infer R`，必须在条件类型中
2. **多个 `infer` 时第一个匹配的被捕获**：同一位置多个 `infer` 取第一个
3. **逆变位置产生交叉类型**：函数参数位置的 `infer` 结果是交叉而非联合
4. **递归 `infer` 可展开嵌套结构**：如 `Awaited` 递归解包多层 Promise
5. **`infer` 约束**：TS 4.7+ 支持 `infer R extends Type` 语法，限制推断结果
6. **性能注意**：深层递归 infer 可能导致编译变慢
