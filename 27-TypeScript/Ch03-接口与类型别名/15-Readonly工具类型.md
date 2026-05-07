# Readonly 工具类型

## 一、概念说明

`Readonly<T>` 将类型 T 的所有属性变为 `readonly`，防止被修改。它是 TypeScript 内置的工具类型，基于映射类型实现。注意这是**浅层只读**，不会递归冻结嵌套对象。

## 二、具体用法

### 2.1 基本用法

```typescript
interface Config {
  host: string;
  port: number;
  debug: boolean;
}

// Readonly<Config> = { readonly host: string; readonly port: number; readonly debug: boolean; }
const config: Readonly<Config> = {
  host: "localhost",
  port: 3000,
  debug: true,
};

// config.host = "other"; // ❌ 编译错误: readonly

console.log(config);
// 输出: { host: "localhost", port: 3000, debug: true }
```

### 2.2 ReadonlyArray

```typescript
const mutableArr = [1, 2, 3];
const readonlyArr: ReadonlyArray<number> = [1, 2, 3];

mutableArr.push(4);    // ✅
// readonlyArr.push(4); // ❌ 编译错误

// readonly 数组不能使用修改方法
// readonlyArr.pop();   // ❌
// readonlyArr[0] = 10; // ❌

console.log(readonlyArr);
// 输出: [1, 2, 3]
```

### 2.3 Readonly 实现原理

```typescript
// Readonly<T> 的源码定义
type MyReadonly<T> = {
  readonly [P in keyof T]: T[P];
};

// 配合 Pick 实现部分只读
type FrozenConfig = Readonly<Pick<Config, "host" | "port">> &
  Pick<Config, "debug">;

const fc: FrozenConfig = {
  host: "localhost",
  port: 3000,
  debug: true,
};
// fc.host = "other"; // ❌ readonly
fc.debug = false;    // ✅ 非只读

console.log(fc.debug);
// 输出: false
```

### 2.4 深 Readonly

```typescript
// 自定义深 Readonly
type DeepReadonly<T> = T extends object
  ? { readonly [K in keyof T]: DeepReadonly<T[K]> }
  : T;

interface Nested {
  a: {
    b: {
      c: string;
    };
  };
}

type FrozenNested = DeepReadonly<Nested>;
// frozen.a.b.c 也是 readonly
```

## 三、注意事项与常见陷阱

1. **浅层只读**：嵌套对象的属性仍可修改
2. **`as const` 更方便**：字面量对象用 `as const` 一步到位
3. **`readonly` 不是 `const`**：`readonly` 用于属性，`const` 用于变量
4. **性能开销极小**：Readonly 只是编译时检查，无运行时开销
