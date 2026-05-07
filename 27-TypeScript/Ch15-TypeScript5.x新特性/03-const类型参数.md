# const类型参数

## 一、概念说明

`const` 类型参数是 TypeScript 5.0 引入的特性，允许泛型函数推断出字面量类型而非拓宽后的类型。在处理配置对象、常量数组等场景中非常有用。

## 二、具体用法

### 2.1 基本用法

```typescript
// 没有 const — 类型被拓宽
function identity<T>(value: T): T {
  return value;
}

const a = identity({ x: 1, y: 'hello' });
// 类型: { x: number; y: string }

// 有 const — 保留字面量类型
function constIdentity<const T>(value: T): T {
  return value;
}

const b = constIdentity({ x: 1, y: 'hello' });
// 类型: { readonly x: 1; readonly y: "hello" }
```

### 2.2 配置对象

```typescript
// 路由配置
function defineRoutes<const T extends Record<string, string>>(routes: T) {
  return routes;
}

const routes = defineRoutes({
  home: '/',
  about: '/about',
  users: '/users/:id',
});
// 类型: { readonly home: "/"; readonly about: "/about"; readonly users: "/users/:id" }

// 可以精确引用
type RoutePath = (typeof routes)[keyof typeof routes];
// "/" | "/about" | "/users/:id"
```

### 2.3 状态管理

```typescript
function defineStore<const T extends { state: object; actions: Record<string, Function> }>(
  config: T
) {
  return config;
}

const store = defineStore({
  state: { count: 0, name: 'counter' },
  actions: {
    increment() { this.state.count++; },
    reset() { this.state.count = 0; },
  },
});
// state 的类型保留字面量类型
```

### 2.4 数组/元组

```typescript
// 没有 const
function toArray<T>(items: T[]): T[] { return items; }
const nums = toArray([1, 2, 3]); // number[]

// 有 const
function toConstArray<const T>(items: T[]): T[] { return items; }
const constNums = toConstArray([1, 2, 3]); // [1, 2, 3]

// 对象数组
const items = toConstArray([
  { id: 1, name: 'A' },
  { id: 2, name: 'B' },
]);
// 类型: readonly [{ readonly id: 1; readonly name: "A" }, ...]
```

### 2.5 联合约束

```typescript
// 使用 const 保留字面量
function getRoutes<const T extends readonly string[]>(routes: T) {
  return routes;
}

const paths = getRoutes(['/home', '/about', '/contact']);
// 类型: readonly ["/home", "/about", "/contact"]
```

## 三、注意事项与常见陷阱

1. **`const` 参数推断出 `readonly` 类型**：属性变为只读
2. **适合配置对象和常量定义**：不需要修改的值
3. **不要用于需要修改的对象**：`readonly` 会阻止修改
4. **可以与 `as const` 效果相同**：但更灵活
5. **TypeScript 5.0+ 支持**
