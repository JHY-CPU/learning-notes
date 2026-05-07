# NoInfer工具类型

## 一、概念说明

`NoInfer` 是 TypeScript 5.4 引入的内置工具类型，用于阻止特定位置的类型推断。它可以让泛型推断更精确，避免推断出过宽的类型。

## 二、具体用法

### 2.1 基本用法

```typescript
// 没有 NoInfer — 问题
function createState<S extends string>(
  initial: S,
  transitions: Record<S, S[]>
): S {
  return initial;
}

// initial 的类型影响 S 的推断
const state = createState('idle', { idle: ['loading'] });
// S 被推断为 "idle"，而不是从 transitions 推断

// 使用 NoInfer — 正确
function createStateSafe<S extends string>(
  initial: NoInfer<S>,
  transitions: Record<S, S[]>
): S {
  return initial;
}

// S 从 transitions 推断
const stateSafe = createStateSafe('idle', { idle: ['loading'], loading: ['success'] });
// S: "idle" | "loading"
```

### 2.2 约束默认值

```typescript
function createEvent<T extends string>(
  name: NoInfer<T>,
  handlers: Record<T, () => void>
): void {
  // ...
}

// 确保 name 必须在 handlers 的键中
createEvent('click', {
  click: () => console.log('clicked'),
  // hover: () => {}, // 可以有额外的
});

// createEvent('unknown', { click: () => {} }); // 错误
```

### 2.3 数组约束

```typescript
function first<T>(arr: NoInfer<T[]>): T | undefined {
  return arr[0];
}

const item = first([1, 2, 3]); // number | undefined
```

### 2.4 默认类型推断

```typescript
type NoInferHelper<T> = [T][T extends any ? 0 : never];

function merge<T extends object>(
  defaults: NoInferHelper<T>,
  overrides: Partial<T>
): T {
  return { ...defaults, ...overrides };
}
```

## 三、注意事项与常见陷阱

1. **`NoInfer` 只阻止推断，不改变类型**
2. **放在需要精确推断的参数位置**
3. **TypeScript 5.4+ 支持**
4. **与手动类型参数不同**：`NoInfer` 仍允许推断，只是不从该位置推断
5. **适合工厂函数和配置函数**
