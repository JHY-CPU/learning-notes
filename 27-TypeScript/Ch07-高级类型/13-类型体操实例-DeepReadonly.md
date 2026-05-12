# 类型体操实例 - DeepReadonly

## 一、概念说明

实现一个递归的 `DeepReadonly<T>` 工具类型，使对象的**所有层级**属性都变为 `readonly`。这是 TypeScript 类型编程中最经典的练习之一，涉及递归类型、条件类型和映射类型的综合运用。与内置 `Readonly<T>`（仅浅层冻结）不同，`DeepReadonly` 递归处理嵌套对象。

## 二、具体用法

### 2.1 基础实现

```typescript
type DeepReadonly<T> = T extends (...args: any[]) => any
  ? T
  : T extends object
    ? { readonly [K in keyof T]: DeepReadonly<T[K]> }
    : T;
```

**实现要点**：
- 先排除函数类型（函数也是 `object`，不应被 readonly 化）
- 对对象类型递归映射每个属性
- 基本类型（string、number 等）直接返回

### 2.2 使用示例

```typescript
interface AppConfig {
  server: {
    host: string;
    port: number;
    ssl: { enabled: boolean; cert: string; key: string };
  };
  database: {
    url: string;
    pool: { min: number; max: number };
  };
  features: string[];
}

type FrozenConfig = DeepReadonly<AppConfig>;

const config: FrozenConfig = {
  server: {
    host: "localhost",
    port: 3000,
    ssl: { enabled: true, cert: "/path/cert", key: "/path/key" },
  },
  database: {
    url: "postgres://localhost/mydb",
    pool: { min: 2, max: 10 },
  },
  features: ["dark-mode", "beta"],
};

// 所有层级都不可修改
// config.server.host = "other";           // 编译错误
// config.server.ssl.enabled = false;      // 编译错误
// config.database.pool.max = 20;          // 编译错误
// config.features.push("new");            // 编译错误
```

### 2.3 排除函数类型

```typescript
// 函数保持不变
type Handler = (event: { type: string }) => void;
type FrozenHandler = DeepReadonly<Handler>;
// 仍然是 (event: { type: string }) => void，不变

// 包含方法的接口
interface Service {
  name: string;
  config: { retries: number };
  execute(): void;
}

type FrozenService = DeepReadonly<Service>;
// name: readonly string
// config: readonly { readonly retries: number }
// execute(): void（函数不变）
```

### 2.4 处理数组

```typescript
// 数组元素也被递归冻结
type ReadonlyConfigList = DeepReadonly<AppConfig[]>;
// readonly DeepReadonly<AppConfig>[]

// 元组类型
type Pair = [{ name: string }, { value: number }];
type FrozenPair = DeepReadonly<Pair>;
// readonly [readonly { readonly name: string }, readonly { readonly value: number }]
```

### 2.5 实际应用：全局常量配置

```typescript
// 定义全局不可变配置
const ROUTES = {
  home: { path: "/", title: "首页", auth: false },
  dashboard: { path: "/dashboard", title: "仪表盘", auth: true },
  settings: {
    path: "/settings",
    title: "设置",
    auth: true,
    children: {
      profile: { path: "/settings/profile", title: "个人资料" },
      security: { path: "/settings/security", title: "安全设置" },
    },
  },
} as const satisfies DeepReadonly<Record<string, object>>;

// typeof ROUTES 已经是深度只读的（as const 保证）
// 但 DeepReadonly 约束确保类型层面的一致性
```

### 2.6 与 JavaScript 的对比

```javascript
// JavaScript 运行时深冻结
function deepFreeze(obj) {
  Object.freeze(obj);
  for (const key of Object.getOwnPropertyNames(obj)) {
    const val = obj[key];
    if (val && typeof val === "object") {
      deepFreeze(val);
    }
  }
  return obj;
}

const frozen = deepFreeze({ a: { b: { c: 1 } } });
frozen.a.b.c = 2; // 静默失败或报错（严格模式）

// TypeScript DeepReadonly：编译时冻结，零运行时开销
```

## 三、注意事项与常见陷阱

1. **函数必须排除**：函数也是对象，递归时先判断 `extends (...args: any[]) => any`，否则函数的属性会被冻结
2. **数组处理**：`DeepReadonly<T[]>` 会产生 `readonly DeepReadonly<T>[]`，可能影响 push/splice 等方法
3. **递归深度限制**：TypeScript 约 1000 层递归上限，过深的嵌套会编译失败
4. **编译性能**：大型复杂类型的 `DeepReadonly` 可能使类型检查变慢数秒
5. **`as const` 可替代**：对于字面量值，`as const` 更简洁，`DeepReadonly` 适合接口/类型别名
6. **`never` 和 `unknown`**：`never` 返回 `never`，`unknown` 视为 object 处理
