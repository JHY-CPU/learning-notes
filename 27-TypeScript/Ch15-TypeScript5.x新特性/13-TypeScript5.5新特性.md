# TypeScript 5.5新特性

## 一、概念说明

TypeScript 5.5 增强了**隐式类型收窄**和**控制流分析**，能更智能地推断类型。特别是 `filter(Boolean)` 后数组类型自动收窄、常量表达式预计算等改进，减少了不必要的类型断言和辅助代码。

## 二、具体用法

### 2.1 filter(Boolean) 自动收窄

```typescript
// 5.5 之前：filtered 类型仍是 (string | null)[]
const items: (string | null)[] = ["a", null, "b", null, "c"];

// 5.5+：filter(Boolean) 自动收窄为 string[]
const nonNull = items.filter(Boolean);
// nonNull: string[]（5.5+）而非 (string | null)[]

// 等价于手写类型守卫
const nonNullManual = items.filter((item): item is string => item !== null);
```

### 2.2 常量表达式预计算

```typescript
// 5.5+：编译时预计算常量表达式
const x = 1 + 2;           // x: 3（之前是 number）
const str = "hello" + " world"; // str: "hello world"（之前是 string）
const flag = true && false;  // flag: false（之前是 boolean）

// 实用：预计算的类型可以用于类型约束
const PORT = 3000 + 1; // PORT: 3001
```

### 2.3 控制流分析增强

```typescript
// 5.5+：更精确的控制流分析
function process(value: string | number): string {
  if (typeof value === "string") {
    return value.toUpperCase();
  }
  // 5.5+ 确保这里 value 一定是 number
  return value.toFixed(2);
}

// 循环中的收窄
function findItem(arr: string[], target: string): string | undefined {
  for (const item of arr) {
    if (item === target) {
      return item; // 类型安全
    }
  }
  // 5.5+ 知道这里没有找到
  return undefined;
}
```

### 2.4 按钮类型守卫

```typescript
// 5.5+：is 判断更智能
function isNotNullish<T>(value: T): value is NonNullable<T> {
  return value !== null && value !== undefined;
}

const items2: (string | null | undefined)[] = ["a", null, undefined, "b"];
const valid = items2.filter(isNotNullish);
// valid: string[]
```

### 2.5 JSDoc 增强

```typescript
// 5.5+：JSDoc 中的类型推断更准确
/**
 * @param {number} x
 * @param {number} y
 * @returns {number}
 */
function add(x: number, y: number): number {
  return x + y;
}
```

### 2.6 与 JavaScript 的对比

```javascript
// JavaScript：filter(Boolean) 工作正常但无类型
const items = ["a", null, "b"];
const nonNull = items.filter(Boolean); // ["a", "b"]
// nonNull 类型仍然是 (string | null)[]

// TypeScript 5.5+：filter(Boolean) 自动收窄类型
// nonNull: string[] — 不需要手动类型断言
```

## 三、注意事项与常见陷阱

1. **`filter(Boolean)` 自动收窄**：TS 5.5+ 特性，之前需要手写类型守卫
2. **控制流分析更精确**：减少了不必要的类型断言和 `!` 非空断言
3. **常量表达式预计算**：`1 + 2` 的类型是 `3` 而非 `number`，`as const` 更有意义
4. **升级后检查类型收窄**：某些旧代码可能依赖之前的"宽松"行为
5. **`filter(Boolean)` 不处理 `0` 和 `""`**：布尔判断会过滤掉 falsy 值，注意使用场景
6. **联合类型收窄更智能**：复杂联合类型的分支处理更精确
