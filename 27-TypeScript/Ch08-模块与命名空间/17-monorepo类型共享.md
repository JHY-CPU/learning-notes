# monorepo 类型共享

## 一、概念说明

在 monorepo（单仓库多包）项目中，多个包之间需要共享类型定义。常见方案包括：**共享类型包**（单独的 `@myapp/types` 包）、**工作区直接引用**（`workspace:*`）、**TypeScript 项目引用**（`references`）。选择合适的方案确保类型一致性和构建效率。

## 二、具体用法

### 2.1 共享类型包

```json
// packages/shared-types/package.json
{
  "name": "@myapp/types",
  "version": "1.0.0",
  "types": "./dist/index.d.ts",
  "main": "./dist/index.js",
  "sideEffects": false
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
  message: string;
};

export type Paginated<T> = {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
};
```

### 2.2 在其他包中使用

```typescript
// packages/api/package.json
{
  "dependencies": {
    "@myapp/types": "workspace:*"
  }
}

// packages/api/src/user.ts
import type { User, ApiResponse, Paginated } from "@myapp/types";

function getUser(id: number): ApiResponse<User> {
  return {
    data: { id, name: "Alice", email: "a@b.com" },
    status: 200,
    message: "OK",
  };
}

function listUsers(page: number): Promise<Paginated<User>> {
  return fetch(`/api/users?page=${page}`).then(r => r.json());
}
```

### 2.3 TypeScript 项目引用

```json
// packages/api/tsconfig.json
{
  "compilerOptions": {
    "composite": true,
    "declaration": true,
    "declarationMap": true
  },
  "references": [
    { "path": "../shared-types" }
  ]
}
```

```bash
# 增量构建
tsc --build

# 只构建变更的包
tsc --build --incremental
```

### 2.4 pnpm 工作区配置

```yaml
# pnpm-workspace.yaml
packages:
  - "packages/*"
  - "apps/*"
```

```json
// packages/api/package.json
{
  "dependencies": {
    "@myapp/types": "workspace:*",
    "@myapp/utils": "workspace:^"
  }
}
```

### 2.5 路径引用（简单场景）

```json
// tsconfig.json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@myapp/types": ["packages/shared-types/src/index"],
      "@myapp/utils": ["packages/utils/src/index"]
    }
  }
}
```

### 2.6 与 JavaScript 的对比

```javascript
// JavaScript monorepo：无类型共享，靠文档和约定
const user = { id: 1, name: "Alice" }; // 不知道其他包期望什么结构

// TypeScript monorepo：类型共享确保一致
// import type { User } from "@myapp/types";
// 所有包使用同一类型定义，编译时发现不一致
```

## 三、注意事项与常见陷阱

1. **类型包只包含类型**：共享类型包通常不需要运行时代码，`sideEffects: false`
2. **工作区链接**：pnpm/yarn/npm workspaces 自动处理本地包链接
3. **构建顺序**：类型包需先构建，使用 `tsc --build` 自动处理依赖顺序
4. **`composite: true`**：启用项目引用必须开启此选项
5. **`workspace:*` vs `workspace:^`**：`*` 匹配任何版本，`^` 遵循 semver
6. **类型包版本管理**：共享类型变更时，需要更新所有引用包的版本
