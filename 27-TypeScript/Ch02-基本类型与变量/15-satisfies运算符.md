# satisfies 运算符

## 一、概念说明

`satisfies` 是 TypeScript 4.9 引入的运算符，用于检查表达式是否满足某个类型约束，**同时保留表达式推断出的精确类型**。与类型断言不同，`satisfies` 不会改变实际类型，只验证类型兼容性。

## 二、具体用法

### 2.1 保留精确类型

```typescript
type Color = "red" | "green" | "blue";
type HexColor = `#${string}`;

// 用 type 注解：丢失了具体 key 的信息
const colors1: Record<Color, HexColor> = {
  red: "#ff0000",
  green: "#00ff00",
  blue: "#0000ff",
};
// colors1 的类型: Record<Color, HexColor>

// 用 satisfies：保留精确类型
const colors2 = {
  red: "#ff0000",
  green: "#00ff00",
  blue: "#0000ff",
} satisfies Record<Color, HexColor>;
// colors2 的类型: { red: "#ff0000"; green: "#00ff00"; blue: "#0000ff" }

// 检查类型
const redColor = colors2.red; // 类型为 "#ff0000"（字面量类型）
console.log(redColor);
// 输出: #ff0000
```

### 2.2 验证配置对象

```typescript
type Route = "/" | "/about" | "/contact";

const routes = {
  "/": () => "首页",
  "/about": () => "关于我们",
  "/contact": () => "联系方式",
} satisfies Record<Route, () => string>;

// 精确类型保留，IDE 能提供更好的补全
console.log(routes["/"]());
// 输出: 首页

// ❌ 缺少路由会报错
// const badRoutes = { "/": () => "首页" } satisfies Record<Route, () => string>;
// 编译错误: Property '"/about"' is missing
```

### 2.3 JSON 数据验证

```typescript
const config = {
  server: {
    port: 3000,
    host: "localhost",
  },
  database: {
    url: "postgres://localhost/mydb",
    pool: 10,
  },
} satisfies {
  server: { port: number; host: string };
  database: { url: string; pool: number };
};

// 保留精确类型，可以安全访问
console.log(config.server.port);
// 输出: 3000
```

## 三、注意事项与常见陷阱

1. **`satisfies` 不改变类型**：只在编译时检查，不影响运行时
2. **比类型注解更精确**：类型注解会"加宽"类型，`satisfies` 保留原始推断
3. **不替代类型注解**：需要"加宽"类型时仍需显式类型注解
4. **TypeScript 4.9+**：旧版本不支持此语法
