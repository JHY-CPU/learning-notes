# 类型导入import-type

## 一、概念说明

`import type` 语法用于只导入类型，编译时会被完全移除，不产生运行时代码。在 `isolatedModules` 模式下是必需的。TS 5.0+ 推荐使用 `verbatimModuleSyntax` 更严格地强制区分值导入和类型导入，优化编译输出的大小和模块加载顺序。

## 二、具体用法

### 2.1 基本语法

```typescript
// 只导入类型（编译后完全消除）
import type { User, CreateUserDto } from "./types";
import type { Request, Response } from "express";

// 导入整个模块的类型
import type * as Types from "./types";
```

### 2.2 内联类型导入（TS 4.5+）

```typescript
// 同一行混合值和类型
import { UserService, type User, type Config } from "./services";

// 等价于
import { UserService } from "./services";
import type { User, Config } from "./services";

// 更简洁，推荐使用
```

### 2.3 verbatimModuleSyntax（TS 5.0+）

```json
// tsconfig.json
{
  "compilerOptions": {
    "verbatimModuleSyntax": true,
    "module": "ESNext",
    "moduleResolution": "Bundler"
  }
}
```

```typescript
// 开启后必须严格区分
import type { User } from "./types";    // 正确：User 是纯类型
import { UserService } from "./service"; // 正确：UserService 是值

// ❌ 如果 User 只有类型定义，用值导入会报错
// import { User } from "./types"; // 错误
```

### 2.4 导出类型

```typescript
// 重新导出类型
export type { User, Config } from "./types";

// 混合导出
export { UserService, API_URL, type User, type Config };

// 全部类型导出
export type * from "./types";
```

### 2.5 枚举的特殊情况

```typescript
// const enum — 值和类型都需要值导入
import { Direction } from "./enums";

// 普通 enum — 类型可以用类型导入
import type { Status } from "./enums";

// 接口/类型别名 — 只用类型导入
import type { User } from "./types";
```

### 2.6 性能影响

```typescript
// ❌ 不必要的值导入（增加 bundle 体积）
import { User, Config, Logger } from "./types";
// 编译后保留 import 语句，即使 User/Config 只是类型

// ✅ 类型导入（编译后消除）
import type { User, Config } from "./types";
import { Logger } from "./types";
// 编译后只保留 Logger 的导入
```

### 2.7 与 JavaScript 的对比

```javascript
// JavaScript：没有类型/值区分
import { User, UserService } from "./module.js";
// 所有导入都产生运行时代码

// TypeScript import type：编译时消除
// import type { User } from "./types"; // 编译后不存在
// import { UserService } from "./service"; // 编译后保留
// 减少不必要的模块依赖
```

## 三、注意事项与常见陷阱

1. **`import type` 不会出现在编译输出中**：减小 bundle 体积，消除运行时依赖
2. **`isolatedModules` 强制使用 `import type`**：跨文件转译器（esbuild、SWC）需要此约定
3. **`verbatimModuleSyntax` 更严格**（TS 5.0+）：推荐新项目使用，比 `isolatedModules` 规范
4. **内联类型导入更简洁**：`import { type X, Y }` 比分两行导入更方便
5. **值和类型混合导入时区分清楚**：接口只有类型，枚举既是值又是类型
6. **Side-effect 导入**：`import "polyfill"` 仍然产生运行时代码，不受 `import type` 影响
