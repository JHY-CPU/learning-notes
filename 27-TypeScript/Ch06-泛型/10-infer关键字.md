# infer 关键字

## 一、概念说明

`infer` 用于在条件类型中**声明推断变量**，从类型结构中提取部分类型。只能在条件类型的 `extends` 子句中使用，是类型推断的核心工具。

## 二、具体用法

### 2.1 提取返回类型

```typescript
type ReturnOf<T> = T extends (...args: any[]) => infer R ? R : never;

type R1 = ReturnOf<() => string>;      // string
type R2 = ReturnOf<(x: number) => boolean>; // boolean
```

### 2.2 提取参数类型

```typescript
type ParamOf<T> = T extends (arg: infer P) => any ? P : never;

type P1 = ParamOf<(x: string) => void>; // string
type P2 = ParamOf<(x: number) => void>; // number
```

### 2.3 提取数组元素类型

```typescript
type ElementOf<T> = T extends (infer E)[] ? E : never;

type E1 = ElementOf<string[]>;          // string
type E2 = ElementOf<[number, boolean]>; // number | boolean
```

### 2.4 提取 Promise 值类型

```typescript
type Awaited<T> = T extends Promise<infer U> ? Awaited<U> : T;

type V = Awaited<Promise<Promise<string>>>; // string（递归展开）
```

## 三、注意事项与常见陷阱

1. **`infer` 只能在条件类型中**：不能单独使用
2. **多个 `infer`**：第一个匹配的 `infer` 被推断
3. **递归 `infer`**：可用于展开嵌套类型
4. **`infer` 与联合类型**：推断结果可能是联合类型
