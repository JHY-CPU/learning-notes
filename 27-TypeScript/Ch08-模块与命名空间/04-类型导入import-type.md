# 类型导入 import type

## 一、概念说明

`import type` 只导入类型信息，编译后完全消除，不产生运行时代码。用于只需要类型检查而不需实际值的场景。

## 二、具体用法

### 2.1 基本类型导入

```typescript
// types.ts
export interface User {
  id: number;
  name: string;
}

export type ID = string | number;

// main.ts
import type { User, ID } from "./types.js"; // 编译后消除

function getUser(id: ID): User {
  return { id: id as number, name: "Alice" };
}

console.log(getUser(1));
// 输出: { id: 1, name: "Alice" }
```

### 2.2 内联类型导入

```typescript
// 同一行中混合值和类型导入
import { UserService, type User } from "./services.js";
```

### 2.3 确保只导入类型

```typescript
// ❌ 不能使用类型导入的值
// import type { helper } from "./utils.js";
// helper(); // 编译错误: 'helper' only refers to a type

// ✅ 需要值时用普通导入
import { helper } from "./utils.js";
```

## 三、注意事项与常见陷阱

1. **消除运行时依赖**：减少打包体积
2. **`isolatedModules`**：推荐开启，确保类型导入一致性
3. **`import { type X }`**：内联语法，TS 4.5+
4. **值和类型同名**：需要同时导入值和类型时用混合语法
