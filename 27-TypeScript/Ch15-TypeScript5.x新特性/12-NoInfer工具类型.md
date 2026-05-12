# NoInfer工具类型

## 一、概念说明

`NoInfer` 是 TypeScript 5.4 引入的内置工具类型，用于**阻止特定位置的类型推断**。在泛型函数中，TypeScript 默认从所有使用类型参数的位置推断类型。`NoInfer` 允许开发者精确控制"从哪里推断"，让泛型推断更精确，避免推断出过宽或不期望的类型。

## 二、具体用法

### 2.1 基本问题和解决方案

```typescript
// 问题：initial 的类型影响推断
function createState<S extends string>(
  initial: S,
  transitions: Record<S, S[]>
) {
  return { initial, transitions };
}

// TypeScript 从 initial "idle" 推断 S 为 "idle"
// transitions 中的其他键被忽略
const state = createState("idle", { idle: ["loading"], loading: ["idle"] });
// S: "idle" — 不够精确

// 解决：使用 NoInfer
function createStateSafe<S extends string>(
  initial: NoInfer<S>,
  transitions: Record<S, S[]>
) {
  return { initial, transitions };
}

// S 从 transitions 推断为 "idle" | "loading"
// initial 必须是 S 的成员
const stateSafe = createStateSafe("idle", { idle: ["loading"], loading: ["idle"] });
// S: "idle" | "loading"
```

### 2.2 事件处理

```typescript
function registerHandler<T extends string>(
  eventName: NoInfer<T>,
  handlers: Record<T, () => void>
): void {
  // 注册逻辑
}

registerHandler("click", {
  click: () => {},
  dblclick: () => {},
  hover: () => {},
});
// T 从 handlers 推断为 "click" | "dblclick" | "hover"
// eventName 必须是其中之一
```

### 2.3 配置合并

```typescript
function mergeConfig<T extends object>(
  defaults: NoInfer<T>,
  overrides: Partial<T>
): T {
  return { ...defaults, ...overrides };
}

const config = mergeConfig(
  { host: "localhost", port: 3000, debug: false },
  { port: 8080, debug: true }
);
// T: { host: string; port: number; debug: boolean }
// overrides 只能包含 T 中的属性
```

### 2.4 数组约束

```typescript
// 不用 NoInfer：arr 的类型被推断为最宽的
function firstBad<T>(arr: T[]): T | undefined {
  return arr[0];
}
const x = firstBad([1, "a", true]); // T: string | number | boolean

// 用 NoInfer：从其他上下文推断
function first<T>(arr: NoInfer<T[]>): T | undefined {
  return arr[0];
}
```

### 2.5 类型守卫

```typescript
function isOneOf<T extends string>(
  value: NoInfer<T>,
  options: readonly T[]
): boolean {
  return options.includes(value);
}

// T 从 options 推断
const valid = isOneOf("GET", ["GET", "POST", "PUT", "DELETE"]);
// T: "GET" | "POST" | "PUT" | "DELETE"
```

### 2.6 手动等效（NoInfer 之前）

```typescript
// TS 5.4 之前的手动方式
type NoInferHelper<T> = [T][T extends any ? 0 : never];

function mergeOld<T extends object>(
  defaults: NoInferHelper<T>,
  overrides: Partial<T>
): T {
  return { ...defaults, ...overrides };
}
```

### 2.7 与 JavaScript 的对比

```javascript
// JavaScript：没有类型推断
function createState(initial, transitions) {
  return { initial, transitions };
}

// TypeScript NoInfer：精确控制推断来源
// 编译时就确保 initial 和 transitions 类型一致
```

## 三、注意事项与常见陷阱

1. **`NoInfer` 只阻止推断，不改变类型**：被 `NoInfer` 包裹的参数仍需满足类型约束
2. **放在"验证"参数而非"数据来源"参数上**：如 `initial: NoInfer<S>` 配合 `transitions: Record<S, S[]>`
3. **TypeScript 5.4+ 支持**：旧版本需使用 `[T][T extends any ? 0 : never]` 手动等效
4. **与手动类型参数不同**：`NoInfer` 仍允许推断，只是不从该位置推断
5. **适合参数有依赖关系的场景**：工厂函数、配置合并、事件注册等
6. **不影响运行时**：纯粹的编译时类型操作
