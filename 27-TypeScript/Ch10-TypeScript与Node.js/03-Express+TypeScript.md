# Express+TypeScript

## 一、概念说明

Express 是最流行的 Node.js Web 框架，结合 TypeScript 使用可以获得路由参数、请求体、响应体的完整类型安全。`@types/express` 提供了所有必要的类型定义。

## 二、具体用法

### 2.1 基本设置

```typescript
import express, { Request, Response, NextFunction } from 'express';

const app = express();
app.use(express.json());

// 基本路由
app.get('/', (req: Request, res: Response) => {
  res.json({ message: 'Hello TypeScript' });
});

app.listen(3000, () => {
  console.log('服务器运行在端口 3000');
});
```

### 2.2 路由参数类型

```typescript
import { Request, Response } from 'express';

// 使用泛型定义请求参数类型
// Request<Params, ResBody, ReqBody, Query>
interface UserParams {
  id: string;
}

interface UserBody {
  name: string;
  email: string;
}

interface UserQuery {
  include?: string;
}

// GET 请求 — 参数和查询类型
app.get(
  '/users/:id',
  (req: Request<UserParams, {}, {}, UserQuery>, res: Response) => {
    const { id } = req.params;   // 类型: string
    const include = req.query.include; // 类型: string | undefined
    res.json({ id, include });
  }
);

// POST 请求 — 请求体类型
app.post(
  '/users',
  (req: Request<{}, {}, UserBody>, res: Response) => {
    const { name, email } = req.body; // 类型: UserBody
    res.status(201).json({ name, email });
  }
);

// PUT 请求
app.put(
  '/users/:id',
  (req: Request<UserParams, {}, UserBody>, res: Response) => {
    const { id } = req.params;
    const { name, email } = req.body;
    res.json({ id, name, email });
  }
);
```

### 2.3 响应类型

```typescript
import { Response } from 'express';

// 通用响应类型
interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

// 辅助函数
function sendSuccess<T>(res: Response, data: T, status = 200) {
  const response: ApiResponse<T> = { success: true, data };
  res.status(status).json(response);
}

function sendError(res: Response, message: string, status = 400) {
  const response: ApiResponse<never> = { success: false, error: message };
  res.status(status).json(response);
}

// 使用
app.get('/users', (req: Request, res: Response) => {
  const users = [{ id: 1, name: '张三' }];
  sendSuccess(res, users);
});
```

### 2.4 路由拆分

```typescript
// routes/users.ts
import { Router, Request, Response } from 'express';

const router = Router();

router.get('/', (req: Request, res: Response) => {
  res.json([{ id: 1, name: '张三' }]);
});

router.get('/:id', (req: Request<{ id: string }>, res: Response) => {
  res.json({ id: req.params.id });
});

export default router;

// server.ts
import userRoutes from './routes/users.js';
app.use('/api/users', userRoutes);
```

## 三、注意事项与常见陷阱

1. **Express 泛型参数顺序**：`Request<Params, ResBody, ReqBody, Query>`
2. **`req.body` 类型默认是 `any`**：必须通过泛型指定或使用验证库
3. **`@types/express` 必须安装**：`npm i -D @types/express`
4. **路由处理函数的返回值类型是 `void`**：不需要 return res
5. **使用验证中间件（如 zod）增强类型安全**：运行时验证 + 编译时类型
