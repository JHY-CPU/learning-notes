# void vs undefined

## 一、概念说明

`void` 和 `undefined` 都可以表示"没有返回值"，但有重要区别：`void` 表示**不使用返回值**（语义化的），`undefined` 是**具体的值**。在函数返回类型中，`void` 更常见，因为它表达了"调用者不应依赖返回值"的意图。

## 二、具体用法

### 2.1 void 返回类型

```typescript
function logMessage(msg: string): void {
  console.log(msg);
  // 可以不写 return，或写 return; 不带值
}

// void 回调：调用者不关心返回值
function forEach(arr: number[], callback: (item: number) => void): void {
  for (const item of arr) {
    callback(item);
  }
}

forEach([1, 2, 3], (item) => {
  console.log(item * 2);
  return item * 2; // 返回值被忽略，不报错
});
// 输出:
// 2
// 4
// 6
```

### 2.2 undefined 返回类型

```typescript
function getUndefined(): undefined {
  // 必须显式 return undefined;
  return undefined;
}

// undefined 要求精确匹配
const val: undefined = getUndefined();
console.log(val);
// 输出: undefined
```

### 2.3 区别对比

```typescript
// void 回调允许返回值（被忽略）
type VoidCallback = () => void;
const fn: VoidCallback = () => 42; // ✅ 允许

// undefined 回调不允许返回值
type UndefinedCallback = () => undefined;
// const fn2: UndefinedCallback = () => 42; // ❌ 编译错误

console.log(fn()); // 输出: 42（返回值存在但被 void 签名忽略）
```

## 三、注意事项与常见陷阱

1. **`void` 回调兼容有返回值的函数**：这是设计特性，方便使用
2. **`undefined` 返回类型严格**：必须精确返回 `undefined`
3. **异步函数**：`Promise<void>` 表示异步操作无返回值
4. **`void` 不是 `undefined`**：虽然 `void` 变量只能赋值 `undefined`，但语义不同
