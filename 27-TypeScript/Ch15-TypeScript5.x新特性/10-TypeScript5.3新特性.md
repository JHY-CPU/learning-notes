# TypeScript 5.3新特性

## 一、概念说明

TypeScript 5.3 带来了 import 属性语法（替代旧的 `assert`）、参数中的 `is` 类型守卫收窄、`switch(true)` 中的类型收窄等改进。Import 属性是 TC39 Stage 3 提案，确保导入模块的类型声明正确。

## 二、具体用法

### 2.1 Import 属性

```typescript
// 使用 with 关键字指定导入类型
import config from "./config.json" with { type: "json" };

// 动态导入
const data = await import("./data.json", { with: { type: "json" } });

// CSS 模块导入
import styles from "./App.css" with { type: "css" };
```

### 2.2 参数中的 is 收窄

```typescript
// 5.3 之前：类型守卫不能作为参数使用
// function process(value, isString: value is string) {} // 错误

// 5.3+：可以在函数参数中使用 is
function processValue(
  value: string | number,
  isString: value is string
) {
  if (isString) {
    // value 被收窄为 string
    console.log(value.toUpperCase());
  } else {
    // value 被收窄为 number
    console.log(value.toFixed(2));
  }
}

// 配合 filter 使用
function filterStrings(arr: (string | number)[]): string[] {
  return arr.filter((item): item is string => typeof item === "string");
}
```

### 2.3 switch(true) 中的类型收窄

```typescript
function check(value: string | number | boolean): string {
  switch (true) {
    case typeof value === "string":
      return value.toUpperCase(); // value: string
    case typeof value === "number":
      return value.toFixed(2);    // value: number
    case typeof value === "boolean":
      return value ? "是" : "否";  // value: boolean
  }
}
```

### 2.4 Import 属性的实际应用

```typescript
// 确保 JSON 导入的类型安全
import schema from "./schema.json" with { type: "json" };

// WASM 模块导入
import wasm from "./module.wasm" with { type: "webassembly" };

// 配合声明文件
declare module "*.json" {
  const value: unknown;
  export default value;
}
```

### 2.5 构造函数类型检查

```typescript
// 5.3+：派生类的构造函数检查更严格
class Base {
  constructor(public value: number) {}
}

class Derived extends Base {
  constructor() {
    super(42); // 必须调用 super
  }
}
```

### 2.6 与 JavaScript 的对比

```javascript
// JavaScript（TC39 提案）：import attributes
import data from "./config.json" with { type: "json" };

// TypeScript 5.3：支持此语法并提供类型检查
// import config from "./config.json" with { type: "json" };
// config 的类型根据声明文件确定
```

## 三、注意事项与常见陷阱

1. **`with { type: "json" }`**：确保 JSON 导入的正确性，替代了旧的 `assert` 语法
2. **参数中的 `is` 收窄**：TS 5.3+ 特性，让类型守卫更灵活
3. **Import 属性是 TC39 Stage 3 提案**：未来可能有变化，但目前稳定
4. **`with` 替代 `assert`**：旧的 `assert { type: "json" }` 语法已废弃
5. **`switch(true)` 收窄**：减少了不必要的类型断言
6. **与旧版本兼容**：如果需要支持 TS 5.3 以下，避免使用新语法
