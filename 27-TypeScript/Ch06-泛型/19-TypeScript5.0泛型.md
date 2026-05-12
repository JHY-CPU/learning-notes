# TypeScript 5.0 泛型

## 一、概念说明

TypeScript 5.0 引入了 `const` 类型参数，允许泛型参数在推断时保持字面量类型而非拓宽为基础类型。这使得泛型函数能推断出更精确的类型，是泛型系统的重要增强。

## 二、具体用法

### 2.1 const 类型参数

```typescript
// 不加 const：args 推断为 (string | number)[]
function fnNormal<T extends readonly unknown[]>(args: T): T {
  return args;
}

// 加 const：args 推断为 readonly [1, 2, 3]
function fn<const T extends readonly unknown[]>(args: T): T {
  return args;
}

const result1 = fnNormal([1, 2, 3]); // (number)[]
const result2 = fn([1, 2, 3]);       // readonly [1, 2, 3]

console.log(result2);
// 输出: readonly [1, 2, 3]
```

### 2.2 与 as const 对比

```typescript
// 之前需要 as const
const arr1 = fn([1, "hello", true] as const);
// 现在 const 参数自动保留字面量
const arr2 = fn([1, "hello", true]);

// 类型相同：readonly [1, "hello", true]
console.log(arr2);
// 输出: [1, "hello", true]
```

### 2.3 配置选项推断

```typescript
function defineConfig<const T extends { name: string; options: readonly string[] }>(
  config: T
): T {
  return config;
}

const config = defineConfig({
  name: "theme",
  options: ["light", "dark", "auto"],
});
// config.options 类型为 readonly ["light", "dark", "auto"]

console.log(config.options);
// 输出: ["light", "dark", "auto"]
```

### 2.4 对象属性冻结

```typescript
function defineRoutes<const T extends Record<string, { method: string; path: string }>>(
  routes: T
): T {
  return routes;
}

const routes = defineRoutes({
  getUser: { method: "GET", path: "/users/:id" },
  createUser: { method: "POST", path: "/users" },
});
// routes.getUser.method 类型为 "GET"（字面量）
// routes.createUser.method 类型为 "POST"（字面量）
```

## 三、注意事项与常见陷阱

1. **只影响推断**：`const` 类型参数改变推断行为，不改变运行时
2. **需要 `extends readonly unknown[]`**：约束数组或元组（对象也适用）
3. **对象也会冻结**：对象属性也会推断为字面量类型
4. **TS 5.0+**：旧版本不支持此语法
5. **不需要 `as const`**：`const` 类型参数替代了部分 `as const` 的使用场景
6. **灵活性权衡**：`const` 参数使类型更精确，但可能过于严格
