# const断言

## 一、概念说明

`as const` 将表达式转换为字面量类型，所有属性变为 `readonly`，数组变为 `readonly` 元组。这是 TypeScript 中创建不可变常量的最简洁方式。

## 二、具体用法

### 2.1 基本用法

```typescript
// 没有 as const
const a = 'hello';        // string
const b = 42;              // number
const c = [1, 2, 3];       // number[]
const d = { x: 1, y: 2 }; // { x: number; y: number }

// 有 as const
const e = 'hello' as const;        // "hello"
const f = 42 as const;              // 42
const g = [1, 2, 3] as const;       // readonly [1, 2, 3]
const h = { x: 1, y: 2 } as const; // { readonly x: 1; readonly y: 2 }
```

### 2.2 路由定义

```typescript
const ROUTES = {
  home: '/',
  about: '/about',
  users: '/users/:id',
} as const;

// 提取路由名
type RouteName = keyof typeof ROUTES; // "home" | "about" | "users"
// 提取路由路径
type RoutePath = (typeof ROUTES)[RouteName]; // "/" | "/about" | "/users/:id"
```

### 2.3 枚举替代

```typescript
const DIRECTIONS = ['up', 'down', 'left', 'right'] as const;
type Direction = (typeof DIRECTIONS)[number]; // "up" | "down" | "left" | "right"

function move(dir: Direction) {
  // ...
}

move('up');    // OK
// move('diagonal'); // 编译错误
```

### 2.4 配置常量

```typescript
const CONFIG = {
  api: {
    baseUrl: 'https://api.example.com',
    timeout: 5000,
    retries: 3,
  },
  features: {
    darkMode: true,
    analytics: false,
  },
} as const;

// 类型完全保留
CONFIG.api.baseUrl // "https://api.example.com"（非 string）
CONFIG.features.darkMode // true（非 boolean）
```

### 2.5 函数返回值

```typescript
// 函数中使用 as const
function getEnv() {
  return {
    nodeEnv: process.env.NODE_ENV ?? 'development',
    port: Number(process.env.PORT ?? '3000'),
  } as const;
}

const env = getEnv();
// 类型: { readonly nodeEnv: string; readonly port: number }
```

### 2.6 解构中的 as const

```typescript
const config = {
  colors: { primary: '#007bff', secondary: '#6c757d' },
  sizes: ['sm', 'md', 'lg'],
};

// 解构时使用 as const
const { colors, sizes } = config as const;
// colors: { readonly primary: "#007bff"; readonly secondary: "#6c757d" }
// sizes: readonly ["sm", "md", "lg"]
```

## 三、注意事项与常见陷阱

1. **`as const` 创建不可变值**：属性变为 `readonly`
2. **适合配置对象和枚举常量**：不需要修改的值
3. **数组变为元组**：`[1, 2] as const` 是 `readonly [1, 2]`
4. **不能与类型断言同时使用**：`as const as string` 不行
5. **函数参数中不需要 `as const`**：使用 `const` 类型参数代替
