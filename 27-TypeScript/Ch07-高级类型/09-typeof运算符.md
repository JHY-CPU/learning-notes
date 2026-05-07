# typeof 运算符

## 一、概念说明

TypeScript 中 `typeof` 有两个用途：**值空间**（运行时检查）和**类型空间**（编译时获取类型）。在类型注解位置使用 `typeof` 可以从值获取其类型。

## 二、具体用法

### 2.1 类型空间的 typeof

```typescript
const config = {
  host: "localhost",
  port: 3000,
  debug: true,
};

// 从值获取类型
type Config = typeof config;
// { host: string; port: number; debug: boolean; }

function useConfig(cfg: Config): void {
  console.log(`${cfg.host}:${cfg.port}`);
}

useConfig(config);
// 输出: localhost:3000
```

### 2.2 配合 ReturnType

```typescript
function createUser(name: string, age: number) {
  return { id: Math.random(), name, age };
}

type User = ReturnType<typeof createUser>;
// { id: number; name: string; age: number; }
```

### 2.3 typeof 在 switch 中

```typescript
function describe(value: string | number | boolean): string {
  switch (typeof value) {
    case "string": return `字符串: ${value}`;
    case "number": return `数字: ${value}`;
    case "boolean": return `布尔: ${value}`;
  }
}

console.log(describe("hello"));
// 输出: 字符串: hello
```

## 三、注意事项与常见陷阱

1. **类型空间 vs 值空间**：`typeof` 在不同位置含义不同
2. **`typeof` 不能用于接口或类型**：只能用于值
3. **`typeof` 获取的是宽泛类型**：`let x = "hello"` 的 `typeof` 是 `string`
4. **`typeof` 结合工具类型**：`ReturnType<typeof fn>` 是常见模式
