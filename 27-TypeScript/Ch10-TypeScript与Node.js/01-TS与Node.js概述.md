# TypeScript与Node.js概述

## 一、概念说明

TypeScript 在 Node.js 后端开发中提供了强大的类型安全能力，从 API 定义到数据库操作都能受益于类型系统。Node.js 生态对 TypeScript 的支持已经非常成熟，有多种运行和编译方案可供选择。

**核心价值**：前后端共享类型定义，端到端类型安全。

## 二、具体用法

### 2.1 运行方案对比

```bash
# 方案一：tsx（推荐，最快）
npm install -D tsx
npx tsx watch src/server.ts

# 方案二：ts-node（经典方案）
npm install -D ts-node typescript @types/node
npx ts-node src/server.ts

# 方案三：先编译再运行（生产环境）
npx tsc && node dist/server.js

# 方案四：Bun 运行（最快启动）
bun run src/server.ts
```

### 2.2 基本 HTTP 服务器

```typescript
import http from 'node:http';

// Request 和 Response 的类型
const server = http.createServer(
  (req: http.IncomingMessage, res: http.ServerResponse) => {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ message: 'Hello TypeScript + Node.js' }));
  }
);

server.listen(3000, () => {
  console.log('服务器运行在 http://localhost:3000');
});
```

### 2.3 前后端类型共享

```typescript
// shared/types.ts — 前后端共享的类型定义
export interface User {
  id: number;
  name: string;
  email: string;
  role: 'admin' | 'user';
}

export interface ApiResponse<T> {
  success: boolean;
  data: T;
  error?: string;
}

export interface PaginatedResponse<T> extends ApiResponse<T[]> {
  total: number;
  page: number;
  pageSize: number;
}
```

### 2.4 Package.json 配置

```json
{
  "scripts": {
    "dev": "tsx watch src/server.ts",
    "build": "tsc",
    "start": "node dist/server.js",
    "type-check": "tsc --noEmit"
  },
  "devDependencies": {
    "typescript": "^5.4.0",
    "tsx": "^4.7.0",
    "@types/node": "^20.11.0"
  }
}
```

## 三、注意事项与常见陷阱

1. **始终安装 `@types/node`**：提供 Node.js 内置模块的类型定义
2. **生产环境用编译后的 JS**：不要用 tsx/ts-node 运行生产代码
3. **`moduleResolution` 使用 `nodenext` 或 `node16`**：匹配 Node.js 的模块解析规则
4. **`type: "module"` 在 package.json 中影响导入**：需要 `.js` 扩展名
5. **ts-node 配置需要 `tsconfig.json` 中的 `ts-node` 字段**：自定义编译选项
