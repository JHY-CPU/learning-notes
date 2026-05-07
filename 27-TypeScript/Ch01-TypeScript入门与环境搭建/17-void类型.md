# void 类型

## 一、概念说明

`void` 表示**没有任何类型**，通常用于表示函数没有返回值。与 `undefined` 不同，`void` 更加语义化——它表达的是"我不关心返回值"。在回调函数和事件处理中广泛使用。

## 二、具体用法

### 2.1 函数返回 void

```typescript
// 不返回任何值的函数
function logMessage(msg: string): void {
  console.log(msg);
  // 没有 return 语句，或 return; 不带值
}

logMessage("Hello TypeScript");
// 输出: Hello TypeScript
```

### 2.2 void 与 undefined 的区别

```typescript
// void 回调：调用者不关心返回值
function forEach(arr: number[], callback: (item: number) => void): void {
  for (const item of arr) {
    callback(item);
  }
}

// 即使回调返回值，也不报错
forEach([1, 2, 3], (item) => {
  console.log(item);
  return item * 2; // 返回值被忽略，但类型兼容
});
// 输出:
// 1
// 2
// 3
```

### 2.3 void 作为类型

```typescript
// void 类型的变量只能赋值为 undefined 或 null（非 strict 模式下）
let voidVar: void;
voidVar = undefined; // ✅

// 在 strictNullChecks 下
// voidVar = null; // ❌ 编译错误
```

## 三、注意事项与常见陷阱

1. **回调中的 void**：声明 `(cb: () => void)` 不阻止回调返回值，只是忽略它
2. **与 `undefined` 的区别**：`void` 更侧重语义，`undefined` 是具体值
3. **Promise\<void\>**：表示异步操作无返回值
4. **不要声明变量为 void**：`void` 类型的变量几乎没有实际用途
