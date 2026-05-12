# readonly 属性

## 一、概念说明

`readonly` 修饰符使类属性在构造函数赋值后不可修改。它提供编译时保护，确保属性值的不可变性。结合构造函数，可以创建不可变的数据对象。`readonly` 是 TypeScript 实现不可变性的重要工具。

## 二、具体用法

### 2.1 基本 readonly

```typescript
class Point {
  readonly x: number;
  readonly y: number;

  constructor(x: number, y: number) {
    this.x = x;
    this.y = y;
  }
}

const point = new Point(10, 20);
console.log(point.x, point.y);
// 输出: 10 20
// point.x = 5; // ❌ 编译错误: Cannot assign to 'x' because it is a read-only property
```

### 2.2 readonly 与默认值

```typescript
class Config {
  readonly version = "1.0.0";
  readonly maxRetries: number;

  constructor(maxRetries: number = 3) {
    this.maxRetries = maxRetries;
  }
}

const config = new Config(5);
console.log(config.version, config.maxRetries);
// 输出: 1.0.0 5
// config.version = "2.0.0"; // ❌ 编译错误
```

### 2.3 readonly 数组和对象

```typescript
class DataManager {
  readonly items: readonly string[];

  constructor(items: string[]) {
    this.items = Object.freeze([...items]);
  }
}

const manager = new DataManager(["a", "b", "c"]);
console.log(manager.items);
// 输出: ["a", "b", "c"]
// manager.items.push("d"); // ❌ readonly 数组
// manager.items = [];       // ❌ readonly 属性
```

### 2.4 readonly 与 as const

```typescript
// 使用 as const 实现更深层的不可变
class ImmutableConfig {
  readonly settings = {
    apiUrl: "https://api.example.com",
    timeout: 5000,
    retries: 3,
  } as const;

  // settings 完全不可变
  // this.settings.apiUrl = "other"; // ❌ readonly
}

const config = new ImmutableConfig();
console.log(config.settings.apiUrl);
// 输出: https://api.example.com
```

### 2.5 与 JavaScript 的对比

```javascript
// JavaScript —— Object.freeze 只冻结浅层
class Config {
  constructor() {
    this.settings = Object.freeze({ host: "localhost", port: 3000 });
  }
}
const c = new Config();
c.settings = {}; // TypeError（严格模式）
c.settings.host = "other"; // ✅ 不报错（浅冻结）
```

```typescript
// TypeScript —— readonly + as const 编译时保护
class Config {
  readonly settings = { host: "localhost", port: 3000 } as const;
}
const c = new Config();
// c.settings = {};      // ❌ 编译错误
// c.settings.host = "other"; // ❌ 编译错误（as const 使属性也 readonly）
```

## 三、注意事项与常见陷阱

1. **`readonly` 只防重新赋值**：不防数组 `push`、对象属性修改（除非用 `as const` 或 `Object.freeze`）
2. **构造函数中可赋值**：这是唯一的赋值时机，之后不可更改
3. **与 `const` 区别**：`readonly` 用于属性，`const` 用于变量
4. **接口中的 readonly**：同理，属性不可修改
5. **readonly 与继承**：子类继承 readonly 属性后仍然是 readonly
6. **组合不可变**：深度不可变需要 `readonly` + `as const` + `Object.freeze`
