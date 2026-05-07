# Node.js TS最佳实践

## 一、概念说明

Node.js + TypeScript 项目需要遵循一套最佳实践来确保代码质量、可维护性和类型安全。从项目结构到类型导出，从错误处理到 API 设计，每个环节都有最佳实践指导。

## 二、具体用法

### 2.1 推荐项目结构

```
project/
├── src/
│   ├── routes/           # 路由层
│   ├── controllers/      # 控制器层
│   ├── services/         # 业务逻辑层
│   ├── models/           # 数据模型
│   ├── middleware/        # 中间件
│   ├── utils/            # 工具函数
│   ├── types/            # 类型定义
│   │   ├── api.ts        # API 类型
│   │   ├── env.d.ts      # 环境变量类型
│   │   └── index.ts      # 类型导出
│   ├── config/           # 配置
│   └── server.ts         # 入口
├── tests/                # 测试
├── tsconfig.json
├── package.json
└── .env
```

### 2.2 类型导出策略

```typescript
// types/index.ts — 统一导出所有类型
export * from './api';
export * from './models';

// 只导出类型（isolatedModules 安全）
export type { User, CreateUserDto, UpdateUserDto } from './models/user';
export type { ApiResponse, PaginatedResponse } from './api';

// 区分值和类型导出
export { AppError, NotFoundError, ValidationError } from './errors'; // 值
export type { ErrorDetails } from './errors'; // 类型
```

### 2.3 严格模式配置

```json
{
  "compilerOptions": {
    "strict": true,
    "noUncheckedIndexedAccess": true,  // 索引访问可能为 undefined
    "noImplicitReturns": true,         // 函数必须显式返回
    "noFallthroughCasesInSwitch": true, // switch 必须 break/return
    "exactOptionalPropertyTypes": true, // 严格区分可选属性
    "noPropertyAccessFromIndexSignature": true // 必须用 [] 访问索引签名
  }
}
```

### 2.4 依赖注入模式

```typescript
// 服务层 — 不直接依赖数据库实例
interface UserRepository {
  findById(id: number): Promise<User | null>;
  create(data: CreateUserDto): Promise<User>;
}

class UserService {
  constructor(private userRepo: UserRepository) {}

  async getUser(id: number): Promise<User> {
    const user = await this.userRepo.findById(id);
    if (!user) throw new NotFoundError('用户', id);
    return user;
  }
}

// 注册依赖
const userRepo = new PostgresUserRepository(pool);
const userService = new UserService(userRepo);
```

### 2.5 错误处理最佳实践

```typescript
// 1. 使用自定义错误类
class AppError extends Error {
  constructor(
    message: string,
    public statusCode: number,
    public code: string
  ) {
    super(message);
  }
}

// 2. 全局未捕获错误处理
process.on('uncaughtException', (err: Error) => {
  logger.fatal('未捕获异常', { error: err.message, stack: err.stack });
  process.exit(1);
});

process.on('unhandledRejection', (reason: unknown) => {
  logger.fatal('未处理的 Promise 拒绝', { reason });
  process.exit(1);
});

// 3. 优雅关闭
function gracefulShutdown(server: Server) {
  return async () => {
    logger.info('收到关闭信号，正在优雅关闭...');
    server.close();
    await db.end();
    process.exit(0);
  };
}

process.on('SIGTERM', gracefulShutdown(server));
```

### 2.6 API 版本管理

```typescript
// 路由版本化
const v1Router = Router();
const v2Router = Router();

app.use('/api/v1', v1Router);
app.use('/api/v2', v2Router);

// 类型版本化
interface UserV1 {
  id: number;
  name: string;
}

interface UserV2 extends UserV1 {
  email: string;
  avatar?: string;
}
```

## 三、注意事项与常见陷阱

1. **始终开启 `strict` 模式**：包括 `noUncheckedIndexedAccess`
2. **类型和值分开导出**：使用 `export type` 确保 `isolatedModules` 兼容
3. **不要用 `any`**：用 `unknown` + 类型守卫代替
4. **使用依赖注入**：提高可测试性和解耦
5. **全局错误处理**：防止未处理的异常导致进程崩溃
