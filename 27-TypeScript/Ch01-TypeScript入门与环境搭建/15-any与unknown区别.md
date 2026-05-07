# any 与 unknown 区别

## 一、概念说明

`any` 和 `unknown` 都表示"任意类型"，但有本质区别：`any` 会**关闭类型检查**，而 `unknown` 是**类型安全的任意类型**。使用 `unknown` 时必须先进行类型收窄才能使用，避免了运行时错误。

## 二、具体用法

### 2.1 any 的行为

```typescript
// any 完全绕过类型检查
let anything: any = "hello";

// 以下操作都不会报错（但可能运行时崩溃）
anything = 42;
anything = { foo: "bar" };
anything.nonExistent.method(); // 编译通过！运行时崩溃

// any 可以赋值给任何类型
const str: string = anything; // ✅ 编译通过
const num: number = anything; // ✅ 编译通过
```

**输出：**
```
编译阶段无报错，但运行时 TypeError: Cannot read properties of undefined
```

### 2.2 unknown 的行为

```typescript
// unknown 是类型安全的
let value: unknown = "hello";

// ❌ 不能直接使用 unknown 的属性或方法
// value.toUpperCase(); // 编译错误！

// ❌ 不能直接赋值给具体类型
// const str: string = value; // 编译错误！

// ✅ 必须先收窄类型
if (typeof value === "string") {
  console.log(value.toUpperCase());
  // 输出: HELLO
}
```

### 2.3 实际应用对比

```typescript
function processData(input: unknown): string {
  // ❌ 不能直接操作
  // return input.toString();

  // ✅ 先检查类型
  if (typeof input === "string") {
    return input.toUpperCase();
  }
  if (typeof input === "number") {
    return input.toFixed(2);
  }
  if (input instanceof Error) {
    return input.message;
  }
  return String(input);
}

console.log(processData("hello"));   // 输出: HELLO
console.log(processData(3.14));      // 输出: 3.14
console.log(processData(new Error("出错"))); // 输出: 出错
```

## 三、注意事项与常见陷阱

1. **`any` 是"类型系统后门"**：它让所有类型检查失效，应尽量避免
2. **`unknown` 是 `any` 的安全替代**：当你不确定类型时，优先用 `unknown`
3. **catch 中的 error**：`catch` 中的错误默认为 `unknown`（strict 模式下）
4. **`unknown` 与 `any` 的赋值关系**：`unknown` 可赋值给 `any`，反之不行
