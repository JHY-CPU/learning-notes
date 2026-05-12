# readonly 映射

## 一、概念说明

通过映射类型添加 `readonly` 修饰符，可以创建不可变类型。内置的 `Readonly<T>` 只做**浅层**冻结，嵌套对象的属性仍可修改。深度 `readonly` 需要递归实现，确保所有层级的属性都变为只读。`-readonly` 修饰符（TS 4.1+）可以反向移除只读约束。

## 二、具体用法

### 2.1 浅层 Readonly

```typescript
type MyReadonly<T> = { readonly [K in keyof T]: T[K] };

interface Config {
  host: string;
  port: number;
}

const config: MyReadonly<Config> = { host: "localhost", port: 3000 };
// config.host = "other"; // 编译错误：只读属性

// 内置 Readonly 等价
const config2: Readonly<Config> = { host: "localhost", port: 3000 };
// config2.host = "other"; // 同样报错
```

### 2.2 深度 Readonly

```typescript
type DeepReadonly<T> = T extends (...args: any[]) => any
  ? T
  : T extends object
    ? { readonly [K in keyof T]: DeepReadonly<T[K]> }
    : T;

interface NestedConfig {
  server: {
    host: string;
    port: number;
    ssl: { enabled: boolean; cert: string };
  };
  db: { url: string };
}

const frozen: DeepReadonly<NestedConfig> = {
  server: { host: "localhost", port: 3000, ssl: { enabled: true, cert: "/cert" } },
  db: { url: "postgres://localhost/db" },
};

// frozen.server.host = "other";      // 错误：深层 readonly
// frozen.server.ssl.enabled = false; // 错误：嵌套属性也 readonly
```

### 2.3 移除 readonly

```typescript
// 浅层移除
type Mutable<T> = { -readonly [K in keyof T]: T[K] };

interface Frozen {
  readonly id: number;
  readonly name: string;
}

type Thawed = Mutable<Frozen>;
// { id: number; name: string }

// 深度移除
type DeepMutable<T> = T extends (...args: any[]) => any
  ? T
  : T extends object
    ? { -readonly [K in keyof T]: DeepMutable<T[K]> }
    : T;

type DeepThawed = DeepMutable<DeepReadonly<NestedConfig>>;
// 恢复所有层级的可写性
```

### 2.4 部分属性只读

```typescript
// 只对特定属性添加 readonly
type ReadonlyKeys<T, K extends keyof T> = T & {
  readonly [P in K]: T[P];
};

interface User {
  id: number;
  name: string;
  email: string;
}

type LockedUser = ReadonlyKeys<User, "id">;
// id 只读，name 和 email 可写
const user: LockedUser = { id: 1, name: "Alice", email: "a@b.com" };
// user.id = 2;    // 错误：只读
user.name = "Bob"; // OK
```

### 2.5 实际应用：不可变配置

```typescript
// 应用启动后配置不可变
type AppConfig = DeepReadonly<{
  api: { baseUrl: string; timeout: number };
  features: { darkMode: boolean; analytics: boolean };
}>;

function createConfig(): AppConfig {
  return {
    api: { baseUrl: "https://api.example.com", timeout: 5000 },
    features: { darkMode: true, analytics: false },
  };
}

const config3 = createConfig();
// config3.api.baseUrl = "other"; // 编译错误
// 确保运行时配置不被意外修改
```

### 2.6 与 JavaScript 的对比

```javascript
// JavaScript：浅层冻结
const config = Object.freeze({ host: "localhost", nested: { a: 1 } });
config.host = "other";       // 静默失败或报错
config.nested.a = 2;         // 可以修改！浅层冻结不保护嵌套

// JavaScript：深层冻结需要递归
function deepFreeze(obj) {
  Object.freeze(obj);
  Object.keys(obj).forEach(key => {
    if (typeof obj[key] === "object" && obj[key] !== null) {
      deepFreeze(obj[key]);
    }
  });
  return obj;
}

// TypeScript DeepReadonly：编译时深层冻结
// 不影响运行时性能，但编译时就能发现赋值错误
```

## 三、注意事项与常见陷阱

1. **`Readonly<T>` 只冻结一层**：嵌套对象属性仍可修改，需用 `DeepReadonly`
2. **深度 Readonly 有编译性能开销**：递归计算复杂嵌套类型可能显著拖慢编译
3. **`-readonly` 需要 TS 4.1+**：旧版本不支持移除 readonly 修饰符
4. **`as const` 是另一种方案**：字面量值用 `as const` 更简单直接
5. **readonly 是编译时保证**：运行时仍可通过类型断言或 `any` 绕过
6. **函数类型应跳过**：递归 `DeepReadonly` 时函数类型不应被 readonly 化
