# ReturnType 与 Parameters

## 一、概念说明

`ReturnType<T>` 提取函数类型的**返回值类型**，`Parameters<T>` 提取函数类型的**参数类型元组**。这两个工具类型用于从已有函数类型中"抽取"信息，是函数式编程和类型推断的重要工具。

## 二、具体用法

### 2.1 ReturnType 基本用法

```typescript
function createUser(name: string, age: number) {
  return { id: Math.random(), name, age, createdAt: new Date() };
}

// 提取返回值类型
type User = ReturnType<typeof createUser>;
// { id: number; name: string; age: number; createdAt: Date; }

const user: User = createUser("Alice", 25);
console.log(user.name);
// 输出: Alice
```

### 2.2 Parameters 基本用法

```typescript
function greet(name: string, greeting: string = "Hello"): string {
  return `${greeting}, ${name}!`;
}

// 提取参数类型元组
type GreetParams = Parameters<typeof greet>;
// [name: string, greeting?: string]

// 使用参数类型
function callWithArgs(...args: GreetParams): string {
  return greet(...args);
}

console.log(callWithArgs("World"));
// 输出: Hello, World!
```

### 2.3 ReturnType 和 Parameters 实现原理

```typescript
// Parameters<T> 源码
type MyParameters<T extends (...args: any) => any> =
  T extends (...args: infer P) => any ? P : never;

// ReturnType<T> 源码
type MyReturnType<T extends (...args: any) => any> =
  T extends (...args: any) => infer R ? R : never;

// infer 关键字用于"推断"类型变量
```

### 2.4 实际应用

```typescript
// 从 API 函数提取响应类型
async function fetchUser(id: number) {
  return { id, name: "Alice", email: "a@b.com" };
}

type UserResponse = Awaited<ReturnType<typeof fetchUser>>;
// { id: number; name: string; email: string; }

function displayUser(user: UserResponse): void {
  console.log(`${user.name} <${user.email}>`);
}

const u: UserResponse = { id: 1, name: "Alice", email: "a@b.com" };
displayUser(u);
// 输出: Alice <a@b.com>
```

## 三、注意事项与常见陷阱

1. **必须用 `typeof`**：`ReturnType<typeof fn>` 而不是 `ReturnType<fn>`
2. **`Awaited<ReturnType<T>>`**：异步函数需用 `Awaited` 展开 Promise
3. **`Parameters` 返回元组**：可以用 `[0]`、`[1]` 等索引访问单个参数类型
4. **泛型函数**：对泛型函数使用时，泛型参数会被推断为约束类型
