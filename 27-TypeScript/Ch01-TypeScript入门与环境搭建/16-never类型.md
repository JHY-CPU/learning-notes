# never 类型

## 一、概念说明

`never` 类型表示**永远不会出现的值**的类型。它用于两种场景：函数永远不会正常返回（总是抛出异常或无限循环），以及在穷尽检查（Exhaustive Checking）中确保所有情况都被处理。`never` 是 TypeScript 类型系统中的"底类型"（Bottom Type），是所有类型的子类型。

## 二、具体用法

### 2.1 抛出异常的函数

```typescript
// 永远抛出错误的函数返回 never
function throwError(message: string): never {
  throw new Error(message);
}

// 永远不会返回的函数
function infiniteLoop(): never {
  while (true) {
    // 无限循环，永远不会结束
  }
}
```

### 2.2 穷尽检查

```typescript
type Shape = "circle" | "square" | "triangle";

function getArea(shape: Shape): number {
  switch (shape) {
    case "circle":
      return Math.PI * 10 * 10;
    case "square":
      return 10 * 10;
    case "triangle":
      return (10 * 10) / 2;
    default:
      // 如果遗漏了某个 case，这里会编译错误
      const _exhaustive: never = shape;
      return _exhaustive;
  }
}

console.log(getArea("circle"));
// 输出: 314.1592653589793
```

### 2.3 联合类型过滤

```typescript
// never 在联合类型中会被自动消除
type A = string | never;       // 等价于 string
type B = string | number | never; // 等价于 string | number

// 条件类型中，never 不会出现在结果中
type NonString<T> = T extends string ? never : T;
type Result = NonString<string | number | boolean>;
// Result = number | boolean（string 被映射为 never 并消除）
```

### 2.4 实际项目中的应用

```typescript
// 安全的类型断言函数
function assertNever(value: never, message?: string): never {
  throw new Error(message ?? `Unexpected value: ${value}`);
}

// API 响应处理
type ApiResponse =
  | { status: "success"; data: unknown }
  | { status: "error"; error: string }
  | { status: "loading" };

function handleResponse(response: ApiResponse): string {
  switch (response.status) {
    case "success":
      return `数据: ${JSON.stringify(response.data)}`;
    case "error":
      return `错误: ${response.error}`;
    case "loading":
      return "加载中...";
    default:
      return assertNever(response); // 如果新增状态未处理，编译报错
  }
}
```

### 2.5 never 与类型收窄

```typescript
function processValue(value: string | number) {
  if (typeof value === "string") {
    console.log(value.toUpperCase());
  } else if (typeof value === "number") {
    console.log(value.toFixed(2));
  } else {
    // 此处 value 的类型为 never
    // 因为 string | number 已被穷尽
    const check: never = value;
  }
}
```

## 三、注意事项与常见陷阱

1. **`never` 没有任何值**：不能将任何值赋给 `never` 类型的变量（除了 `never` 本身）
2. **与 `void` 的区别**：`void` 表示没有返回值，`never` 表示永远不会返回
3. **穷尽检查是最佳实践**：当处理联合类型时，用 `never` 确保不遗漏情况
4. **空数组类型**：空数组 `[]` 的类型推断为 `never[]`
5. **函数返回类型推断**：TypeScript 会自动将抛出异常的函数推断为 `never`
6. **`never` 作为交集的单位元**：`T & never` 等价于 `never`
