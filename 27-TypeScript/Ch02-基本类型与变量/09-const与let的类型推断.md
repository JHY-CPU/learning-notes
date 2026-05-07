# const 与 let 的类型推断

## 一、概念说明

`const` 和 `let` 在类型推断上有本质区别：`const` 推断为**字面量类型**（最窄类型），而 `let` 推断为**基础类型**（宽泛类型）。这是因为 `const` 变量不可重新赋值，所以 TypeScript 可以安全地推断出精确的字面量类型。

## 二、具体用法

### 2.1 基础推断差异

```typescript
// const 推断为字面量类型
const constStr = "hello"; // 类型: "hello"（字面量类型）
const constNum = 42;      // 类型: 42（字面量类型）
const constBool = true;   // 类型: true（字面量类型）

// let 推断为基础类型
let letStr = "hello";     // 类型: string
let letNum = 42;          // 类型: number
let letBool = true;       // 类型: boolean

// 验证类型
const check1: "hello" = constStr; // ✅
// const check2: "hello" = letStr; // ❌ string 不能赋给 "hello"
```

### 2.2 对象 const 推断

```typescript
// const 对象：属性仍然可变，推断为基础类型
const obj = {
  name: "Alice",
  age: 25,
};
// 类型: { name: string; age: number }

obj.name = "Bob"; // ✅ 属性可以修改

// as const 冻结所有属性为字面量类型
const frozenObj = {
  name: "Alice",
  age: 25,
} as const;
// 类型: { readonly name: "Alice"; readonly age: 25 }

// frozenObj.name = "Bob"; // ❌ 只读

console.log(frozenObj.name);
// 输出: Alice
```

### 2.3 数组 const 推断

```typescript
const mutableArr = [1, 2, 3];       // 类型: number[]
const readonlyArr = [1, 2, 3] as const; // 类型: readonly [1, 2, 3]

mutableArr.push(4);    // ✅
// readonlyArr.push(4); // ❌ readonly 数组

console.log(readonlyArr);
// 输出: readonly [1, 2, 3]
```

## 三、注意事项与常见陷阱

1. **`const` 不保证不可变**：`const` 对象的属性仍可修改，只是变量引用不可变
2. **`as const` 深度冻结**：使用 `as const` 后所有层级变为 `readonly`
3. **字面量类型导致类型过窄**：某些场景下 `const` 推断可能过于严格
4. **性能无差异**：`const` 和 `let` 在运行时行为一致，仅类型层面不同
