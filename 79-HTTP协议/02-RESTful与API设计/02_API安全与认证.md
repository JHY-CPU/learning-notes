# API 安全与认证


## API 安全与认证


JWTOAuth 2.0HMAC


API 安全是 Web 应用安全的核心，包括认证（你是谁）、授权（你能做什么）、数据完整性和防攻击措施。


## JWT（JSON Web Token）


```
JWT 结构：
  Header.Payload.Signature
  三部分用 . 分隔，每部分 Base64Url 编码

Header（头部）：
  {
    "alg": "HS256",     // 签名算法
    "typ": "JWT"         // 令牌类型
  }

Payload（载荷）：
  {
    // 标准声明（Registered Claims）
    "sub": "1234567890",     // 主题（用户ID）
    "name": "张三",          // 自定义声明
    "role": "admin",
    "iat": 1698765432,       // 签发时间
    "exp": 1698769032,       // 过期时间
    "iss": "my-api",         // 签发者
    "aud": "my-app"          // 受众
  }

Signature（签名）：
  HMACSHA256(
    base64UrlEncode(header) + "." + base64UrlEncode(payload),
    secret
  )

使用方式：
  // 登录获取 Token
  POST /api/auth/login
  {"username": "zhangsan", "password": "123456"}
  → {"token": "eyJhbGciOiJIUzI1...", "refresh_token": "..."}

  // 请求时携带 Token
  GET /api/v1/users/123
  Authorization: Bearer eyJhbGciOiJIUzI1...

JWT 最佳实践：
  1. 设置过期时间（access_token 15-30分钟）
  2. 使用 refresh_token 刷新（7-30天）
  3. 敏感信息不要放入 Payload（Base64 可解码）
  4. 使用 HTTPS 传输
  5. 签名密钥足够复杂（256位以上）
  6. 支持 Token 吊销（黑名单机制）

Token 刷新流程：
  ┌─────────────────────────────────────────────┐
  │  客户端                  服务端              │
  │    │── 登录 ────────────→│                   │
  │    │← access_token(15m) ─│                   │
  │    │← refresh_token(7d)──│                   │
  │    │                      │                   │
  │    │── API请求 + token ──→│                   │
  │    │← 200 ───────────────│                   │
  │    │                      │                   │
  │    │── API请求 + token ──→│                   │
  │    │← 401 Unauthorized ──│ (token过期)       │
  │    │                      │                   │
  │    │── POST /refresh ────→│                   │
  │    │    + refresh_token   │                   │
  │    │← 新 access_token ───│                   │
  └─────────────────────────────────────────────┘

Token 黑名单：
  // 用户登出时，将 token 加入黑名单
  SET blacklist:token:{jti} 1 EX 1800
  // 过期时间 = token 剩余有效期

  // 每次验证 token 时检查黑名单
  if redis.exists(f"blacklist:token:{jti}"):
      raise Unauthorized("Token 已失效")
```


## OAuth 2.0 授权框架


```
OAuth 2.0 四种授权模式：

1. 授权码模式（Authorization Code）— 最安全
  ┌─────────────────────────────────────────────┐
  │  用户 → 客户端 → 授权服务器 → 资源服务器    │
  │                                             │
  │  1. 用户点击"使用微信登录"                   │
  │  2. 跳转授权页面                            │
  │     GET /authorize?response_type=code        │
  │       &client_id=xxx                         │
  │       &redirect_uri=https://app.com/callback │
  │       &scope=read:user                       │
  │       &state=abc123                          │
  │  3. 用户同意授权                            │
  │  4. 重定向到 callback，携带 code             │
  │     https://app.com/callback?code=AUTH_CODE  │
  │  5. 客户端用 code 换 token                   │
  │     POST /token                              │
  │     grant_type=authorization_code            │
  │     code=AUTH_CODE                           │
  │     client_id=xxx&client_secret=yyy          │
  │  6. 返回 access_token + refresh_token        │
  └─────────────────────────────────────────────┘

2. 隐式模式（Implicit）— 不推荐
  // 直接返回 token（无 code 交换）
  // 适合纯前端应用，但安全性低
  // 已被 PKCE + 授权码模式替代

3. 密码模式（Password）— 内部系统
  POST /token
  grant_type=password
  username=zhangsan&password=123456
  client_id=xxx&client_secret=yyy
  // 用户名密码直接给客户端，仅限信任的内部应用

4. 客户端模式（Client Credentials）— 机器间通信
  POST /token
  grant_type=client_credentials
  client_id=xxx&client_secret=yyy
  // 没有用户参与，服务间调用

PKCE（Proof Key for Code Exchange）：
  // 公开客户端（移动端/SPA）的安全增强
  // 防止授权码被拦截

  1. 客户端生成随机 code_verifier
  2. 计算 code_challenge = SHA256(code_verifier)
  3. 授权请求带上 code_challenge
  4. 换 token 时带上 code_verifier
  5. 服务端验证 SHA256(code_verifier) == code_challenge

Scope 权限控制：
  scope=read:user      // 读取用户信息
  scope=write:user     // 修改用户信息
  scope=read:orders    // 读取订单
  scope=admin          // 管理员权限

  // Token 中包含已授权的 scope
  {
    "access_token": "...",
    "scope": "read:user read:orders",
    "expires_in": 3600
  }
```


## API Key 与 HMAC 签名


```
API Key 认证：
  // 简单的 API Key
  GET /api/v1/users
  X-API-Key: ak_1234567890abcdef

  // 或通过查询参数（不推荐，可能被日志记录）
  GET /api/v1/users?api_key=ak_1234567890abcdef

  API Key 管理：
  - 每个客户端分配唯一 Key
  - 支持 Key 轮换（生成新 Key，旧 Key 过渡期有效）
  - 记录 Key 使用日志
  - 支持权限绑定（Scope）

HMAC 签名认证：
  // 请求参数 + 时间戳 + 密钥 → 签名
  // 防止请求被篡改和重放

签名过程：
  1. 构造签名字符串
     string_to_sign = "POST\n"
                    + "/api/v1/orders\n"
                    + "access_key=ak_xxx\n"
                    + "timestamp=1698765432\n"
                    + "nonce=a1b2c3d4\n"
                    + body_hash

  2. 计算 HMAC-SHA256
     signature = HMAC-SHA256(string_to_sign, secret_key)

  3. 请求头
     Authorization: HMAC-SHA256
       access_key=ak_xxx,
       timestamp=1698765432,
       nonce=a1b2c3d4,
       signature=base64(signature)

服务端验证：
  1. 检查 timestamp（5分钟内有效，防重放）
  2. 检查 nonce（已使用则拒绝，防重放）
  3. 用相同方式构造签名字符串
  4. 用密钥计算 HMAC
  5. 比较签名是否一致

防重放机制：
  // Redis 存储已使用的 nonce
  SET nonce:{nonce} 1 EX 300
  // 如果已存在 → 拒绝请求

  // 时间戳验证
  if abs(server_time - request_timestamp) > 300:
      reject("请求已过期")

签名示例（Python）：
  import hmac, hashlib, time

  def sign_request(method, path, body, access_key, secret_key):
      timestamp = str(int(time.time()))
      nonce = generate_nonce()
      body_hash = hashlib.sha256(body.encode()).hexdigest()

      string_to_sign = f"{method}\n{path}\n{access_key}\n{timestamp}\n{nonce}\n{body_hash}"
      signature = hmac.new(secret_key.encode(), string_to_sign.encode(), hashlib.sha256).hexdigest()

      return {"Authorization": f"HMAC-SHA256 access_key={access_key}, timestamp={timestamp}, nonce={nonce}, signature={signature}"}
```


## 限流（Rate Limiting）


```
常见限流算法：

1. 固定窗口计数器：
  // 每分钟最多100次请求
  时间窗口：00:00 - 01:00 → 计数器++
  超过100 → 拒绝
  问题：窗口边界可能有突发流量

2. 滑动窗口计数器：
  // 精确控制任意1分钟内的请求数
  当前时间 - 60s 内的请求数 < 100 → 允许
  用 Redis sorted set 实现

  ZADD rate:user:123 {timestamp} {request_id}
  ZREMRANGEBYSCORE rate:user:123 0 {now - 60s}
  count = ZCARD rate:user:123
  if count >= 100: reject()

3. 令牌桶算法：
  // 每秒产生10个令牌，桶容量100
  ┌──────────────────────────────┐
  │  令牌桶                      │
  │  ████████████░░ (80/100)    │
  │  每请求消耗1个令牌           │
  │  每秒补充10个令牌（最多100） │
  │  桶空 → 拒绝请求            │
  └──────────────────────────────┘

4. 漏桶算法：
  // 固定速率处理请求
  ┌──────────────────────────────┐
  │  请求 → [桶] → 固定速率流出 │
  │  桶满 → 拒绝新请求          │
  │  输出速率恒定               │
  └──────────────────────────────┘

Nginx 限流配置：
  # 定义限流区域
  limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

  server {
      location /api/ {
          limit_req zone=api burst=20 nodelay;
          # burst=20 允许突发20个请求
          # nodelay 突发请求不延迟立即处理
      }
  }

限流响应：
  HTTP/1.1 429 Too Many Requests
  Retry-After: 60
  X-RateLimit-Limit: 100
  X-RateLimit-Remaining: 0
  X-RateLimit-Reset: 1698765492

  {"error": "Rate limit exceeded", "retry_after": 60}

多级限流：
  // 全局限流：10000次/分钟
  // 用户级限流：100次/分钟
  // 接口级限流：POST /orders → 10次/分钟
  // IP级限流：1000次/分钟
```


## 输入验证与安全防护


```
输入验证：
  // 服务端永远不要信任客户端输入
  // 所有输入都必须验证

  // 类型验证
  if not isinstance(user_id, int):
      reject("user_id must be integer")

  // 长度验证
  if len(username) > 50 or len(username) < 1:
      reject("username length must be 1-50")

  // 格式验证
  if not re.match(r'^[a-zA-Z0-9_]+$', username):
      reject("username contains invalid characters")

  // 范围验证
  if age < 0 or age > 150:
      reject("invalid age")

常见攻击与防护：

SQL 注入：
  // 错误：字符串拼接
  f"SELECT * FROM users WHERE id = {user_id}"
  // 攻击：user_id = "1 OR 1=1"

  // 正确：参数化查询
  cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))

XSS（跨站脚本）：
  // API 响应中设置安全头
  Content-Security-Policy: default-src 'self'
  X-Content-Type-Options: nosniff
  X-Frame-Options: DENY

  // 输出编码
  username = html.escape(user_input)

CSRF（跨站请求伪造）：
  // 使用 CSRF Token
  // 或检查 Origin/Referer 头
  // 或使用 SameSite Cookie

CORS 配置：
  Access-Control-Allow-Origin: https://myapp.com
  Access-Control-Allow-Methods: GET, POST, PUT, DELETE
  Access-Control-Allow-Headers: Content-Type, Authorization
  Access-Control-Max-Age: 86400
  // 不要使用 * 通配符（生产环境）

HTTPS 强制：
  // HTTP 严格传输安全
  Strict-Transport-Security: max-age=31536000; includeSubDomains

  // Nginx 配置
  server {
      listen 80;
      return 301 https://$host$request_uri;
  }

安全清单：
  ┌─────────────────────────────────────────────┐
  │  □ 所有输入验证（类型/长度/格式/范围）     │
  │  □ HTTPS 强制（HSTS）                      │
  │  □ 认证和授权（JWT/OAuth）                 │
  │  □ 限流和防刷                              │
  │  □ CORS 白名单                             │
  │  □ 安全响应头                              │
  │  □ SQL 注入防护（参数化查询）              │
  │  □ 日志审计                                │
  │  □ 密钥安全存储                            │
  │  □ 敏感数据脱敏                            │
  └─────────────────────────────────────────────┘
```


> **Note:** API 安全的核心是认证（JWT/OAuth 2.0）、授权（Scope/角色）、数据完整性（HMAC 签名）和防攻击（限流/输入验证）。JWT 适合无状态认证，OAuth 2.0 适合第三方授权，HMAC 适合服务间通信。限流推荐令牌桶算法，输入验证永远在服务端执行。


<!-- Converted from: 02_API安全与认证.html -->
