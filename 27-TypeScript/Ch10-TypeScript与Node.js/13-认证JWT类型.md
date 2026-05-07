# 认证JWT类型

## 一、概念说明

JWT（JSON Web Token）认证是 Node.js 后端最常见的认证方案。TypeScript 可以精确地类型化 token 的 payload、认证中间件扩展的 Request 对象，以及各种认证状态。

## 二、具体用法

### 2.1 JWT Payload 类型

```typescript
import jwt from 'jsonwebtoken';

// JWT Payload 类型
interface JwtPayload {
  userId: number;
  role: 'admin' | 'user';
  iat?: number;  // issued at（自动添加）
  exp?: number;  // expiration（自动添加）
}

// 生成 token
function generateToken(payload: Omit<JwtPayload, 'iat' | 'exp'>): string {
  return jwt.sign(payload, process.env.JWT_SECRET, {
    expiresIn: '7d',
  });
}

// 验证 token
function verifyToken(token: string): JwtPayload {
  return jwt.verify(token, process.env.JWT_SECRET) as JwtPayload;
}
```

### 2.2 扩展 Express Request

```typescript
// types/express.d.ts
declare global {
  namespace Express {
    interface Request {
      user?: {
        userId: number;
        role: 'admin' | 'user';
      };
    }
  }
}

// 认证中间件
import { RequestHandler } from 'express';

const authenticate: RequestHandler = (req, res, next) => {
  const authHeader = req.headers.authorization;

  if (!authHeader?.startsWith('Bearer ')) {
    res.status(401).json({ error: '未提供认证 token' });
    return;
  }

  const token = authHeader.split(' ')[1];

  try {
    const decoded = verifyToken(token);
    req.user = { userId: decoded.userId, role: decoded.role };
    next();
  } catch (err) {
    res.status(401).json({ error: 'token 无效或已过期' });
  }
};

// 授权中间件
const authorize = (...roles: Array<'admin' | 'user'>): RequestHandler => {
  return (req, res, next) => {
    if (!req.user) {
      res.status(401).json({ error: '未认证' });
      return;
    }

    if (!roles.includes(req.user.role)) {
      res.status(403).json({ error: '权限不足' });
      return;
    }

    next();
  };
};
```

### 2.3 使用认证中间件

```typescript
// 路由中使用
app.get('/profile', authenticate, (req, res) => {
  // req.user 类型是 { userId: number; role: 'admin' | 'user' }
  res.json({ userId: req.user!.userId, role: req.user!.role });
});

app.delete('/users/:id', authenticate, authorize('admin'), (req, res) => {
  // 只有 admin 可以删除用户
  res.json({ deleted: req.params.id });
});
```

### 2.4 Refresh Token 类型

```typescript
interface TokenPair {
  accessToken: string;
  refreshToken: string;
  expiresIn: number;
}

interface StoredRefreshToken {
  token: string;
  userId: number;
  expiresAt: Date;
  deviceInfo?: string;
}

// 刷新 token
async function refreshAccessToken(refreshToken: string): Promise<TokenPair> {
  const stored = await db.refreshTokens.findOne({ token: refreshToken });

  if (!stored || stored.expiresAt < new Date()) {
    throw new UnauthorizedError('refresh token 无效或已过期');
  }

  const user = await db.users.findById(stored.userId);

  const newAccessToken = generateToken({
    userId: user.id,
    role: user.role,
  });

  return {
    accessToken: newAccessToken,
    refreshToken: stored.token,
    expiresIn: 7 * 24 * 60 * 60, // 7天
  };
}
```

### 2.5 Cookie 认证类型

```typescript
import cookieParser from 'cookie-parser';

app.use(cookieParser());

interface AuthCookies {
  accessToken?: string;
  refreshToken?: string;
}

app.get('/protected', (req, res) => {
  const cookies = req.cookies as AuthCookies;
  // cookies.accessToken 类型是 string | undefined
});
```

## 三、注意事项与常见陷阱

1. **JWT Secret 必须在环境变量中**：不要硬编码
2. **`jwt.verify` 同步版本可能抛异常**：使用 try-catch 包裹
3. **Request 扩展需要 `.d.ts` 文件**：确保被 tsconfig 包含
4. **Token 刷新要防重放攻击**：每个 refresh token 只能用一次
5. **不要在 payload 中放敏感信息**：JWT 可以被解码（虽然不能被篡改）
