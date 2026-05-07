# readonly 映射

## 一、概念说明

通过映射类型添加 `readonly` 修饰符，可以创建深度只读类型。`Readonly<T>` 是浅层的，深度 `readonly` 需要递归实现。

## 二、具体用法

### 2.1 浅层 Readonly

```typescript
type MyReadonly<T> = { readonly [K in keyof T]: T[K] };

interface Config { host: string; port: number; }
const config: MyReadonly<Config> = { host: "localhost", port: 3000 };
// config.host = "other"; // ❌ readonly
```

### 2.2 深度 Readonly

```typescript
type DeepReadonly<T> = T extends object
  ? { readonly [K in keyof T]: DeepReadonly<T[K]> }
  : T;

interface Nested {
  a: { b: { c: string } };
}

const obj: DeepReadonly<Nested> = { a: { b: { c: "hello" } } };
// obj.a.b.c = "world"; // ❌ 深层 readonly
```

### 2.3 移除 readonly

```typescript
type Mutable<T> = { -readonly [K in keyof T]: T[K] };
type DeepMutable<T> = T extends object
  ? { -readonly [K in keyof T]: DeepMutable<T[K]> }
  : T;
```

## 三、注意事项与常见陷阱

1. **`Readonly<T>` 是浅层的**：嵌套对象属性仍可修改
2. **深度 Readonly 有性能开销**：递归计算复杂类型
3. **`-readonly` 移除修饰符**：TS 4.1+ 支持
4. **`as const` 是另一种方案**：字面量用 `as const` 更简单
