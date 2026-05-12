# infer 关键字

## 一、概念说明

`infer` 用于在条件类型中**声明推断变量**，从类型结构中提取部分类型。只能在条件类型的 `extends` 子句中使用，是类型推断的核心工具。`infer` 让 TypeScript 能从复杂类型中"解构"出需要的部分。

## 二、具体用法

### 2.1 提取返回类型

```typescript
type ReturnOf<T> = T extends (...args: any[]) => infer R ? R : never;

type R1 = ReturnOf<() => string>;      // string
type R2 = ReturnOf<(x: number) => boolean>; // boolean
type R3 = ReturnOf<typeof Math.random>; // number
```

### 2.2 提取参数类型

```typescript
type ParamOf<T> = T extends (arg: infer P) => any ? P : never;

type P1 = ParamOf<(x: string) => void>; // string
type P2 = ParamOf<(x: number) => void>; // number

// 提取所有参数
type Params<T> = T extends (...args: infer P) => any ? P : never;
type AllParams = Params<(a: string, b: number) => void>; // [string, number]
```

### 2.3 提取数组元素类型

```typescript
type ElementOf<T> = T extends (infer E)[] ? E : never;

type E1 = ElementOf<string[]>;          // string
type E2 = ElementOf<[number, boolean]>; // number | boolean
type E3 = ElementOf<Array<Date>>;       // Date
```

### 2.4 提取 Promise 值类型

```typescript
type Awaited<T> = T extends Promise<infer U> ? Awaited<U> : T;

type V = Awaited<Promise<Promise<string>>>; // string（递归展开）
type V2 = Awaited<Promise<number>>;         // number
type V3 = Awaited<string>;                  // string
```

### 2.5 实际应用

```typescript
// 提取 Map 的值类型
type MapValue<T> = T extends Map<any, infer V> ? V : never;
type V = MapValue<Map<string, number>>; // number

// 提取对象某个属性的类型
type PropType<T, K extends keyof T> = T extends { [P in K]: infer V } ? V : never;
type NameType = PropType<{ name: string; age: number }, "name">; // string

// 提取类的构造函数参数
type ConstructorParams<T> = T extends new (...args: infer P) => any ? P : never;
type UserParams = ConstructorParams<typeof User>; // 如果 User 有 constructor(name: string, age: number) => [string, number]
```

## 三、注意事项与常见陷阱

1. **`infer` 只能在条件类型中**：不能单独使用，必须在 `extends` 子句中
2. **多个 `infer`**：同一个位置的第一个匹配的 `infer` 被推断
3. **递归 `infer`**：可用于展开嵌套类型，如 `Promise<Promise<T>>`
4. **`infer` 与联合类型**：推断结果可能是联合类型
5. **`infer` 的位置**：可以出现在返回类型、参数类型、数组元素类型等位置
6. **`infer R` 必须在分支中使用**：推断的 R 只能在条件类型的 true 分支中引用
