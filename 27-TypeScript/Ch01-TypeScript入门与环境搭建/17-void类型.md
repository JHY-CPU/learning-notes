# void 类型

## 一、概念说明

`void` 表示**没有任何类型**，通常用于表示函数没有返回值。与 `undefined` 不同，`void` 更加语义化——它表达的是"我不关心返回值"。在回调函数和事件处理中广泛使用。理解 `void` 与 `undefined` 的区别是 TypeScript 类型系统的基础知识。

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

### 2.4 Promise\<void\> 的使用

```typescript
// 异步操作无返回值
async function saveLog(message: string): Promise<void> {
  await fetch("/api/logs", {
    method: "POST",
    body: JSON.stringify({ message }),
  });
  // 不需要返回值
}

// 调用者只需等待完成
async function main() {
  await saveLog("用户登录");
  console.log("日志已保存");
}
```

### 2.5 void 在事件处理中的应用

```typescript
// 事件处理器类型
type EventHandler = (event: Event) => void;

// 不同的处理器实现都可以兼容
const clickHandler: EventHandler = (e) => {
  console.log("点击事件");
};

const submitHandler: EventHandler = (e) => {
  e.preventDefault();
  console.log("表单提交");
  return false; // 返回值被 void 忽略，但类型兼容
};

// 数组方法中的 void
const numbers = [1, 2, 3, 4, 5];
numbers.forEach((n) => console.log(n)); // 回调返回 void
numbers.forEach((n) => n * 2); // 回调返回 number，也被 void 接受
```

### 2.6 void 与 never 的对比

```typescript
// void：函数正常执行完毕，只是没有返回值
function doNothing(): void {
  // 正常结束
}

// never：函数永远不会正常结束
function crash(): never {
  throw new Error("崩溃");
}

// void 可以赋值给 void
const a: void = doNothing(); // ✅

// never 可以赋值给任何类型
const b: string = crash(); // ✅ never 是所有类型的子类型
```

## 三、注意事项与常见陷阱

1. **回调中的 void**：声明 `(cb: () => void)` 不阻止回调返回值，只是忽略它
2. **与 `undefined` 的区别**：`void` 更侧重语义，`undefined` 是具体值
3. **Promise\<void\>**：表示异步操作无返回值，需 `await` 或 `.then()` 处理
4. **不要声明变量为 void**：`void` 类型的变量几乎没有实际用途
5. **`this:void` 模式**：用于声明函数不依赖 `this`，常见于回调签名
6. **严格模式下的 void**：`strictNullChecks` 开启后，`void` 变量只能赋 `undefined`
