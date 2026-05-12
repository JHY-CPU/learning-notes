# TypeScript 5.4新特性

## 一、概念说明

TypeScript 5.4 引入了 `NoInfer` 工具类型和闭包中的类型收窄保留。`NoInfer` 可以阻止特定位置的类型推断，让泛型推断更精确。闭包中保留类型收窄减少了 `!` 非空断言的使用。

## 二、具体用法

### 2.1 NoInfer 工具类型

```typescript
// 没有 NoInfer — initial 的值影响推断
function createStateBad<S extends string>(
  initial: S,
  transitions: Record<S, S[]>
) {
  return { initial, transitions };
}

// S 被推断为 "idle"（从 initial 推断），而不是从 transitions 推断
createStateBad("idle", { idle: ["loading"] }); // S: "idle"

// 使用 NoInfer — 只从 transitions 推断
function createState<S extends string>(
  initial: NoInfer<S>,
  transitions: Record<S, S[]>
) {
  return { initial, transitions };
}

// S 从 transitions 推断，initial 只能是 S 的成员
createState("idle", { idle: ["loading"], loading: ["success"] });
// S: "idle" | "loading"
```

### 2.2 闭包中的类型收窄保留

```typescript
// 5.4 之前：闭包中 items 可能失去收窄
function processOld(items: string[] | null) {
  if (items === null) return;

  setTimeout(() => {
    // items 可能仍然是 string[] | null
    // 需要 items!.length 或重复检查
  }, 100);
}

// 5.4+：闭包中 items 保持 string[] 类型
function processNew(items: string[] | null) {
  if (items === null) return;

  setTimeout(() => {
    console.log(items.length); // items: string[]（收窄保留）
  }, 100);
}
```

### 2.3 NoInfer 实战：配置函数

```typescript
function createEvent<T extends string>(
  name: NoInfer<T>,
  handlers: Record<T, () => void>
): void {
  // name 必须在 handlers 的键中
}

createEvent("click", {
  click: () => console.log("clicked"),
  hover: () => console.log("hovered"),
});
// createEvent("unknown", { click: () => {} }); // 错误
```

### 2.4 NoInfer 约束默认值

```typescript
function merge<T extends object>(
  defaults: NoInfer<T>,
  overrides: Partial<T>
): T {
  return { ...defaults, ...overrides };
}

const config = merge(
  { host: "localhost", port: 3000, debug: false },
  { port: 8080 }
);
// config: { host: string; port: number; debug: boolean }
```

### 2.5 多参数推断优先级

```typescript
// 使用 NoInfer 指定从哪个参数推断
function assign<T extends object>(
  target: NoInfer<T>,
  source: T
): T {
  return Object.assign({}, target, source);
}

// T 从 source 推断，target 只需兼容
const result = assign(
  { a: 1, b: 2 },
  { a: 10, b: 20, c: 30 }
);
// T: { a: number; b: number; c: number }
```

### 2.6 与 JavaScript 的对比

```javascript
// JavaScript：没有类型推断的概念
function createState(initial, transitions) {
  return { initial, transitions };
}

// TypeScript 5.4 NoInfer：精确控制推断来源
// 编译时确保 initial 必须在 transitions 的键中
```

## 三、注意事项与常见陷阱

1. **`NoInfer` 只阻止推断，不改变类型**：它告诉 TS 不要从某个位置推断类型参数
2. **放在需要精确推断的参数位置**：通常是"输入验证"参数而非"数据来源"参数
3. **TS 5.4+ 支持**：旧版本不支持 `NoInfer`
4. **闭包收窄保留**：减少了 `!` 非空断言的使用，代码更安全
5. **与手动类型参数不同**：`NoInfer` 仍允许推断，只是不从特定位置推断
6. **适合工厂函数和配置函数**：参数之间有依赖关系时非常有用
