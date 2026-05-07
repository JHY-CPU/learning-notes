# infer 推断详解

## 一、概念说明

`infer` 在条件类型中声明类型变量，从复杂类型中"提取"子类型。它是类型模式匹配的核心工具。

## 二、具体用法

### 2.1 提取函数类型信息

```typescript
type Return<T> = T extends (...args: any[]) => infer R ? R : never;
type Args<T> = T extends (...args: infer P) => any ? P : never;

type R = Return<(x: number) => string>; // string
type P = Args<(x: number, y: string) => void>; // [number, string]
```

### 2.2 提取 Promise 内部类型

```typescript
type Awaited<T> = T extends Promise<infer U> ? Awaited<U> : T;

type R1 = Awaited<Promise<string>>;           // string
type R2 = Awaited<Promise<Promise<number>>>;  // number
```

### 2.3 提取数组元素

```typescript
type Elem<T> = T extends (infer E)[] ? E : never;
type E1 = Elem<string[]>; // string
type E2 = Elem<[number, boolean]>; // number | boolean
```

### 2.4 提取对象值类型

```typescript
type ValueOf<T> = T[keyof T];
type V = ValueOf<{ a: string; b: number }>; // string | number
```

## 三、注意事项与常见陷阱

1. **多个 `infer`**：第一个匹配的被推断
2. **逆变位置**：函数参数的 `infer` 结果是交叉类型
3. **递归 `infer`**：可用于展开嵌套结构
4. **`infer` 只能用在 `extends` 子句中**
