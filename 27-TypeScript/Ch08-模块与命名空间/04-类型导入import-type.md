# 类型导入 import type

## 一、概念说明

`import type` 只导入类型信息，编译后**完全消除**，不产生运行时代码。它用于只需要类型检查而不需实际值的场景，能减小打包体积、消除不必要的运行时依赖。在 `isolatedModules` 和 `verbatimModuleSyntax` 模式下，类型导入是必需的。

## 二、具体用法

### 2.1 基本类型导入

```typescript
// types.ts
export interface User {
  id: number;
  name: string;
  email: string;
}

export type ID = string | number;
export type Status = "active" | "inactive";

// main.ts — 只导入类型
import type { User, ID, Status } from "./types.js";
// 编译后这行完全消除，不产生 require/import 语句

function getUser(id: ID): User {
  return { id: id as number, name: "Alice", email: "a@b.com" };
}
```

### 2.2 内联类型导入（TS 4.5+）

```typescript
// 同一行中混合值和类型导入
import { UserService, type User, type Config } from "./services.js";

// 等价于两条导入
import { UserService } from "./services.js";
import type { User, Config } from "./services.js";
```

### 2.3 类型导入的限制

```typescript
import type { helper } from "./utils.js";

// ❌ 不能使用类型导入的值
// helper(); // 编译错误: 'helper' only refers to a type

// ✅ 需要值时用普通导入
import { helper } from "./utils.js";
helper(); // OK

// ✅ 类型和值都需要时，混合导入
import { helper, type HelperOptions } from "./utils.js";
```

### 2.4 verbatimModuleSyntax（TS 5.0+）

```json
// tsconfig.json
{
  "compilerOptions": {
    "verbatimModuleSyntax": true
  }
}
```

```typescript
// 开启后必须严格区分值导入和类型导入
import type { User } from "./types";    // 正确：User 是纯类型
import { UserService } from "./services"; // 正确：UserService 是值

// ❌ 错误：User 只有类型定义，不能用值导入
// import { User } from "./types";
```

### 2.5 导出类型

```typescript
// 重新导出类型
export type { User, Config } from "./types.js";

// 混合导出
export { UserService, API_URL, type User, type Config };
```

### 2.6 与 JavaScript 的对比

```javascript
// JavaScript：没有类型/值的区别
import { User, UserService } from "./module.js";
// 所有导入都会产生运行时代码

// TypeScript：区分类型导入
import { UserService } from "./module.js";    // 运行时存在
import type { User } from "./types.js";      // 编译后消除
// 减少不必要的运行时依赖和 bundle 体积
```

## 三、注意事项与常见陷阱

1. **消除运行时依赖**：`import type` 不影响打包体积和模块加载顺序
2. **`isolatedModules` 强制使用**：开启此选项后，纯类型导入必须用 `import type`
3. **`verbatimModuleSyntax` 更严格**（TS 5.0+）：推荐新项目使用，比 `isolatedModules` 更规范
4. **内联语法更简洁**：`import { type X, Y }` 比分两行导入更方便
5. **值和类型同名**：枚举（enum）既是值又是类型，需要值导入；接口只有类型
6. **Side-effect 与类型导入**：`import type "module"` 不会执行模块代码
