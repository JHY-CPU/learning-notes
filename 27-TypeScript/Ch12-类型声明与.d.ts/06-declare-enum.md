# declare enum

## 一、概念说明

`declare enum` 声明全局枚举的类型。由于 `.d.ts` 文件不能包含实现，声明的枚举只提供类型信息，不会生成运行时的枚举对象。

## 二、具体用法

### 2.1 基本枚举声明

```typescript
// 全局枚举
declare enum LogLevel {
  Debug,
  Info,
  Warn,
  Error,
}

// 带值的枚举
declare enum HttpStatus {
  OK = 200,
  NotFound = 404,
  InternalServerError = 500,
}

// 字符串枚举
declare enum Direction {
  Up = 'UP',
  Down = 'DOWN',
  Left = 'LEFT',
  Right = 'RIGHT',
}
```

### 2.2 const enum 声明

```typescript
// const enum — 编译时内联，性能更好
declare const enum Environment {
  Development = 'development',
  Production = 'production',
  Test = 'test',
}

// 使用时会被内联替换
const env = Environment.Development; // 编译后: const env = 'development';
```

### 2.3 模块中的枚举

```typescript
declare module 'my-lib' {
  export enum Status {
    Pending = 'PENDING',
    Active = 'ACTIVE',
    Closed = 'CLOSED',
  }

  export function getStatus(): Status;
}

// 使用
import { Status, getStatus } from 'my-lib';
const s = getStatus(); // Status
if (s === Status.Active) { /* ... */ }
```

### 2.4 枚举与联合类型

```typescript
declare enum Theme {
  Light = 'light',
  Dark = 'dark',
  System = 'system',
}

// 枚举成员的类型
type ThemeValue = `${Theme}`; // 'light' | 'dark' | 'system'

// 用作 props 类型
interface AppConfig {
  theme: Theme; // 只能是 Theme 的成员
}
```

## 三、注意事项与常见陷阱

1. **`declare enum` 不能有实现**：只有成员声明
2. **`declare const enum` 需要 `isolatedModules` 兼容处理**：可能不被 esbuild 等工具支持
3. **枚举值需要明确赋值**：没有初始值的成员会被推断为 `number`
4. **推荐用联合类型替代枚举**：`type Status = 'active' | 'inactive'` 更简洁
5. **枚举在声明文件中需要命名空间声明**：否则可能无法正确引用
