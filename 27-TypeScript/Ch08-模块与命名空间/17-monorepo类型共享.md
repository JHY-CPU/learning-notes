# monorepo 类型共享

## 一、概念说明

在 monorepo（单仓库多包）项目中，多个包之间需要共享类型定义。常见方案包括：**工作区类型包**、**内部包直接引用**、**路径引用**。

## 二、具体用法

### 2.1 工作区类型包

```json
// packages/shared-types/package.json
{
  "name": "@myapp/types",
  "version": "1.0.0",
  "types": "./dist/index.d.ts",
  "main": "./dist/index.js"
}
```

```typescript
// packages/shared-types/src/index.ts
export interface User {
  id: number;
  name: string;
  email: string;
}

export type ApiResponse<T> = {
  data: T;
  status: number;
};
```

### 2.2 在其他包中使用

```typescript
// packages/api/src/user.ts
import type { User, ApiResponse } from "@myapp/types";

function getUser(id: number): ApiResponse<User> {
  return {
    data: { id, name: "Alice", email: "a@b.com" },
    status: 200,
  };
}

console.log(getUser(1).data.name);
// 输出: Alice
```

### 2.3 TypeScript 项目引用

```json
// packages/api/tsconfig.json
{
  "references": [
    { "path": "../shared-types" }
  ],
  "compilerOptions": {
    "composite": true
  }
}
```

## 三、注意事项与常见陷阱

1. **类型包只包含类型**：不需要运行时代码
2. **工作区链接**：pnpm/npm/yarn workspaces 自动链接本地包
3. **构建顺序**：类型包需先构建
4. **`composite: true`**：启用项目引用需开启此选项
