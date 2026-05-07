# 测试Node+TS

## 一、概念说明

Node.js + TypeScript 的测试首选 Vitest（或 Jest），两者都原生支持 TypeScript。测试配置需要正确设置类型声明、模块解析和模拟（Mock）的类型支持。

## 二、具体用法

### 2.1 Vitest 配置（推荐）

```typescript
// vitest.config.ts
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
    },
    include: ['src/**/*.test.ts'],
  },
});

// package.json
// { "scripts": { "test": "vitest", "test:run": "vitest run", "coverage": "vitest run --coverage" } }
```

### 2.2 基本测试类型

```typescript
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { UserService } from './user.service';

describe('UserService', () => {
  let service: UserService;

  beforeEach(() => {
    service = new UserService();
  });

  it('应该创建用户', async () => {
    const user = await service.create({
      name: '张三',
      email: 'zhangsan@example.com',
    });

    expect(user).toMatchObject({
      id: expect.any(Number),
      name: '张三',
      email: 'zhangsan@example.com',
    });
  });

  it('应该拒绝重复邮箱', async () => {
    await service.create({ name: '张三', email: 'test@example.com' });

    await expect(
      service.create({ name: '李四', email: 'test@example.com' })
    ).rejects.toThrow('邮箱已存在');
  });
});
```

### 2.3 Mock 函数类型

```typescript
import { vi, Mock } from 'vitest';

// Mock 数据库
interface MockDb {
  users: {
    findById: Mock<[id: number], Promise<User | null>>;
    create: Mock<[data: CreateUserDto], Promise<User>>;
    delete: Mock<[id: number], Promise<void>>;
  };
}

function createMockDb(): MockDb {
  return {
    users: {
      findById: vi.fn(),
      create: vi.fn(),
      delete: vi.fn(),
    },
  };
}

// 测试中使用
const mockDb = createMockDb();
const service = new UserService(mockDb);

it('应该查找用户', async () => {
  mockDb.users.findById.mockResolvedValue({ id: 1, name: '张三' });

  const user = await service.findById(1);

  expect(user).toEqual({ id: 1, name: '张三' });
  expect(mockDb.users.findById).toHaveBeenCalledWith(1);
});
```

### 2.4 Mock 模块

```typescript
// Mock 外部模块
vi.mock('./database', () => ({
  query: vi.fn(),
  connect: vi.fn(),
}));

import { query } from './database';

it('应该执行查询', async () => {
  vi.mocked(query).mockResolvedValue({ rows: [{ id: 1 }] });

  const result = await getUserById(1);

  expect(query).toHaveBeenCalledWith(
    expect.stringContaining('SELECT'),
    [1]
  );
});
```

### 2.5 测试 Express 路由

```typescript
import request from 'supertest';
import { app } from '../app';

describe('GET /api/users', () => {
  it('应该返回用户列表', async () => {
    const response = await request(app)
      .get('/api/users')
      .set('Authorization', 'Bearer test-token')
      .query({ page: 1 });

    expect(response.status).toBe(200);
    expect(response.body.success).toBe(true);
    expect(response.body.data).toBeInstanceOf(Array);
  });

  it('未认证应该返回 401', async () => {
    const response = await request(app).get('/api/users');
    expect(response.status).toBe(401);
  });
});
```

### 2.6 集成测试类型

```typescript
import { beforeAll, afterAll } from 'vitest';
import { setup, teardown } from './test-utils';

let testDb: TestDatabase;

beforeAll(async () => {
  testDb = await setup(); // 启动测试数据库
});

afterAll(async () => {
  await teardown(testDb); // 清理
});
```

## 三、注意事项与常见陷阱

1. **Vitest 比 Jest 对 TS 支持更好**：原生 ESM、更快的执行速度
2. **`vi.mocked()` 获得正确的 mock 类型**：比手动类型断言更安全
3. **测试文件放在 `src/` 目录内**：确保被 tsconfig 的 `include` 覆盖
4. **`supertest` 需要安装 `@types/supertest`**：类型定义不在内置包中
5. **Mock 的类型应与原函数一致**：确保测试的有效性
