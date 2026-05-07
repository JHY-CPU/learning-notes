# TypeScript 5.5新特性

## 一、概念说明

TypeScript 5.5 增强了隐式类型收窄和控制流分析，能更智能地推断类型。

## 二、具体用法

### 2.1 隐式类型收窄增强

```typescript
// 5.5+：数组 filter 后类型自动收窄
const items: (string | null)[] = ['a', null, 'b', null];

// 5.5 之前：filtered 类型仍是 (string | null)[]
// 5.5+：filtered 类型是 string[]
const filtered = items.filter((item): item is string => item !== null);

// even 更简洁
const nonNull = items.filter(Boolean); // 5.5+：string[]
```

### 2.2 控制流分析

```typescript
// 5.5+：更精确的控制流分析
function process(value: string | number) {
  if (typeof value === 'string') {
    return value.toUpperCase(); // string
  }
  return value.toFixed(2); // number — 5.5+ 确保这里一定不是 string
}

// 循环中的收窄
function findItem(arr: string[], target: string) {
  for (const item of arr) {
    if (item === target) {
      return item; // 类型安全
    }
  }
  // 5.5+ 知道这里没有找到
}
```

### 2.3 预计算常量表达式

```typescript
// 5.5+：编译时预计算
const x = 1 + 2; // 类型: 3（之前是 number）
const str = 'hello' + ' world'; // 类型: "hello world"
```

## 三、注意事项与常见陷阱

1. **`filter(Boolean)` 自动收窄**：5.5+ 特性
2. **控制流分析更精确**：减少不必要的类型断言
3. **常量表达式预计算**：`as const` 更有用
4. **升级后检查类型收窄是否正确**
