# Bun+TypeScript

## 一、概念说明

Bun 是一个高性能的 JavaScript/TypeScript 运行时，原生支持 TypeScript，无需额外配置。Bun 内置了打包器、测试运行器和包管理器，提供了一体化的开发体验。

## 二、具体用法

### 2.1 基本使用

```bash
# 安装 Bun
curl -fsSL https://bun.sh/install | bash

# 运行 TypeScript 文件 — 直接执行，无需编译
bun run server.ts

# 安装依赖（比 npm 快 25x）
bun install

# 运行测试
bun test
```

### 2.2 HTTP 服务器

```typescript
// server.ts — Bun 原生 HTTP
const server = Bun.serve({
  port: 3000,
  fetch(req: Request): Response | Promise<Response> {
    const url = new URL(req.url);

    if (url.pathname === '/api/hello') {
      return Response.json({ message: 'Hello from Bun!' });
    }

    return new Response('Not Found', { status: 404 });
  },
});

console.log(`服务器运行在 http://localhost:${server.port}`);
```

### 2.3 文件与 IO

```typescript
// Bun 内置了优化的文件 API
const file = Bun.file('data.json');

// 读取文件
const content: string = await file.text();
const json: unknown = await file.json();
const buffer: ArrayBuffer = await file.arrayBuffer();

// 写入文件
await Bun.write('output.txt', 'Hello Bun');

// 文件是否存在
const exists: boolean = await file.exists();
```

### 2.4 SQLite（内置）

```typescript
import { Database } from 'bun:sqlite';

const db = new Database('app.db');

// 创建表
db.run(`
  CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL
  )
`);

// 类型化的查询
interface User {
  id: number;
  name: string;
  email: string;
}

const stmt = db.prepare<User, [string]>('SELECT * FROM users WHERE name = ?');
const user = stmt.get('张三'); // User | null
```

### 2.5 测试

```typescript
import { describe, it, expect } from 'bun:test';

describe('计算器', () => {
  it('应该正确相加', () => {
    expect(1 + 1).toBe(2);
  });

  it('应该处理异步', async () => {
    const result = await Promise.resolve(42);
    expect(result).toBe(42);
  });
});
```

### 2.6 tsconfig 配置

```json
{
  "compilerOptions": {
    "target": "ESNext",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "types": ["bun-types"],
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true
  }
}
```

```bash
# 安装 Bun 类型定义
bun add -d bun-types
```

## 三、注意事项与常见陷阱

1. **Bun 原生支持 TS**：不需要 ts-node 或 tsx
2. **`bun-types` 包**：提供 Bun 特有 API 的类型定义
3. **Bun 不完全兼容 Node.js API**：某些模块可能不可用
4. **Bun 的 `fetch` 和 `Request` 类型与 Node.js 不同**：使用全局的 Web API 类型
5. **生产环境仍需测试兼容性**：Bun 还在快速迭代中
