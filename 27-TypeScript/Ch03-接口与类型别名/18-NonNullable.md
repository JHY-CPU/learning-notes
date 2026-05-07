# NonNullable

## 一、概念说明

`NonNullable<T>` 从类型 T 中**排除 `null` 和 `undefined`**，返回非空类型。它是 TypeScript 内置的工具类型，等价于 `Exclude<T, null | undefined>`。在 `strictNullChecks` 模式下非常有用。

## 二、具体用法

### 2.1 基本用法

```typescript
type MaybeString = string | null | undefined;

// 排除 null 和 undefined
type DefinitelyString = NonNullable<MaybeString>;
// string

const s: DefinitelyString = "hello"; // ✅
// const n: DefinitelyString = null;  // ❌ 编译错误

console.log(s);
// 输出: hello
```

### 2.2 NonNullable 实现原理

```typescript
// NonNullable<T> 源码
type MyNonNullable<T> = Exclude<T, null | undefined>;
// 等价于 T extends null | undefined ? never : T

// 应用于对象属性
interface User {
  name: string | null;
  email: string | undefined;
  age: number;
}

type RequiredUser = {
  [K in keyof User]: NonNullable<User[K]>;
};
// { name: string; email: string; age: number; }
```

### 2.3 配合其他工具类型

```typescript
// 从函数返回类型中排除 null
function findUser(id: number): { name: string } | null {
  return id === 1 ? { name: "Alice" } : null;
}

type UserResult = NonNullable<ReturnType<typeof findUser>>;
// { name: string }（排除了 null）

const user: UserResult = { name: "Alice" };
console.log(user);
// 输出: { name: "Alice" }
```

### 2.4 数组中的 NonNullable

```typescript
type MixedArray = (string | null | undefined)[];
type CleanArray = NonNullable<MixedArray[number]>[];
// string[]

const clean: CleanArray = ["a", "b", "c"];
// clean.push(null); // ❌

console.log(clean);
// 输出: ["a", "b", "c"]
```

## 三、注意事项与常见陷阱

1. **只排除 null 和 undefined**：不排除其他假值如 `""`、`0`、`false`
2. **配合 strictNullChecks**：非严格模式下 null 可赋给任何类型
3. **浅层排除**：不递归排除嵌套对象中的 null
4. **性能无开销**：纯编译时类型操作
