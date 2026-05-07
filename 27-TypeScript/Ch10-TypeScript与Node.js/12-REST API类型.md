# REST API类型

## 一、概念说明

REST API 的类型安全涵盖路由参数、查询参数、请求体和响应体。通过定义统一的 API 类型，可以在前后端之间共享类型契约，确保数据格式一致。

## 二、具体用法

### 2.1 API 类型定义

```typescript
// types/api.ts — 统一 API 类型定义

// 通用响应
interface ApiResponse<T = unknown> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

// 分页参数
interface PaginationQuery {
  page?: number;
  pageSize?: number;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
}

// 分页响应
interface PaginatedResponse<T> extends ApiResponse<T[]> {
  total: number;
  page: number;
  pageSize: number;
  totalPages: number;
}

// API 路由类型
interface ApiRoutes {
  'GET /users': { query: PaginationQuery; response: PaginatedResponse<User> };
  'GET /users/:id': { params: { id: string }; response: ApiResponse<User> };
  'POST /users': { body: CreateUserDto; response: ApiResponse<User> };
  'PUT /users/:id': { params: { id: string }; body: UpdateUserDto; response: ApiResponse<User> };
  'DELETE /users/:id': { params: { id: string }; response: ApiResponse<null> };
}
```

### 2.2 Express 路由实现

```typescript
import { Request, Response } from 'express';

// 查询参数类型
interface UserQuery extends PaginationQuery {
  search?: string;
  role?: 'admin' | 'user';
  active?: boolean;
}

// 请求体类型
interface CreateUserDto {
  name: string;
  email: string;
  password: string;
  role?: 'admin' | 'user';
}

interface UpdateUserDto {
  name?: string;
  email?: string;
  role?: 'admin' | 'user';
}

// 路由处理
app.get('/users',
  async (req: Request<{}, {}, {}, UserQuery>, res: Response<ApiResponse<User[]>>) => {
    const { page = 1, pageSize = 20, search, role } = req.query;

    const users = await userService.findMany({
      page: Number(page),
      pageSize: Number(pageSize),
      search,
      role,
    });

    res.json({ success: true, data: users });
  }
);

app.post('/users',
  async (req: Request<{}, {}, CreateUserDto>, res: Response<ApiResponse<User>>) => {
    const user = await userService.create(req.body);
    res.status(201).json({ success: true, data: user });
  }
);
```

### 2.3 前端 API 客户端类型

```typescript
// 前端类型化 API 客户端
class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  async get<T>(path: string, query?: Record<string, unknown>): Promise<ApiResponse<T>> {
    const url = new URL(path, this.baseUrl);
    if (query) {
      Object.entries(query).forEach(([k, v]) => {
        if (v !== undefined) url.searchParams.set(k, String(v));
      });
    }

    const res = await fetch(url);
    return res.json();
  }

  async post<T>(path: string, body: unknown): Promise<ApiResponse<T>> {
    const res = await fetch(`${this.baseUrl}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    return res.json();
  }
}

// 使用
const api = new ApiClient('http://localhost:3000/api');
const { data: users } = await api.get<User[]>('/users', { page: 1 });
```

### 2.4 Zod 验证请求体

```typescript
import { z } from 'zod';

const createUserSchema = z.object({
  name: z.string().min(2).max(50),
  email: z.string().email(),
  password: z.string().min(8),
  role: z.enum(['admin', 'user']).default('user'),
});

type CreateUserDto = z.infer<typeof createUserSchema>;

// 验证中间件
function validate<T>(schema: z.ZodSchema<T>): RequestHandler {
  return (req, res, next) => {
    const result = schema.safeParse(req.body);
    if (!result.success) {
      res.status(400).json({
        success: false,
        error: result.error.format(),
      });
      return;
    }
    req.body = result.data;
    next();
  };
}

app.post('/users', validate(createUserSchema), createUserHandler);
```

## 三、注意事项与常见陷阱

1. **DTO 类型和数据库类型分开定义**：避免泄露数据库 schema
2. **使用 Zod 做运行时验证**：TypeScript 类型检查只在编译时
3. **查询参数都是 `string`**：需要手动转换类型
4. **统一响应格式**：所有 API 返回相同的结构
5. **共享类型放在 `shared/` 目录**：前后端都能访问
