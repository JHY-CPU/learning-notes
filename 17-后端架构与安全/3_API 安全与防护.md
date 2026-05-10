# API 安全与防护


## 🛡️ API 安全与防护


CORS 配置、CSRF 防护、速率限制 (令牌桶/滑动窗口)、安全头 (CSP/HSTS/X-Frame-Options)、Helmet.js、API Key 管理。


## CORS (跨域资源共享)


```
// ========== CORS 是什么 ==========
// 浏览器安全策略: 默认禁止跨域请求
// CORS 告诉浏览器允许哪些源访问

// ========== CORS 请求头 ==========
// 响应头:
// Access-Control-Allow-Origin: https://example.com
// Access-Control-Allow-Methods: GET,POST,PUT,DELETE
// Access-Control-Allow-Headers: Content-Type,Authorization
// Access-Control-Allow-Credentials: true
// Access-Control-Max-Age: 3600

// 预检请求 (OPTIONS):
// Method: OPTIONS
// Origin: https://example.com
// Access-Control-Request-Method: POST

// ========== Go CORS 中间件 ==========
func CORSMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        origin := c.GetHeader("Origin")

        // 白名单验证
        allowedOrigins := map[string]bool{
            "https://example.com":    true,
            "https://admin.example.com": true,
        }

        if allowedOrigins[origin] {
            c.Header("Access-Control-Allow-Origin", origin)
            c.Header("Access-Control-Allow-Credentials", "true")
        }

        c.Header("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS")
        c.Header("Access-Control-Allow-Headers", "Content-Type,Authorization,X-Request-ID")
        c.Header("Access-Control-Max-Age", "3600")

        if c.Request.Method == "OPTIONS" {
            c.AbortWithStatus(204)
            return
        }

        c.Next()
    }
}

// ========== CORS 安全建议 ==========
// ❌ 危险: Access-Control-Allow-Origin: *
// ✅ 安全: 白名单特定 origin
// ✅ 避免在 production 使用 *
// ✅ 需要凭据时不能用 *
// ✅ 限制允许的方法和头
```


## CSRF (跨站请求伪造)


```
// ========== CSRF 攻击 ==========
// 用户已登录银行网站
// 攻击者诱导用户访问恶意网站
// 恶意网站自动提交转账请求
// 浏览器自动携带用户的 cookie!

// ========== CSRF 防护 ==========
// 1. CSRF Token (主流)
//    服务器生成 token, 嵌入表单
//    提交时验证 token

// 2. SameSite Cookie
//    Set-Cookie: session=xxx; SameSite=Strict
//    Strict: 完全禁止跨站发送
//    Lax: 允许部分 GET (默认)
//    None: 允许跨站 (需 Secure)

// 3. Origin/Referer 验证
//    检查请求头中的 Origin

// 4. 自定义请求头
//    X-Requested-With: XMLHttpRequest
//    浏览器不会自动发送自定义头

// ========== Go CSRF 中间件 ==========
// 使用 gorilla/csrf 或自行实现

// import "github.com/gorilla/csrf"

// func setupCSRF(r *gin.Engine) {
//     // 生成 CSRF 中间件
//     csrfMiddleware := csrf.Protect(
//         []byte("32-byte-long-auth-key"),
//         csrf.Secure(true),   // HTTPS 才发送
//         csrf.Path("/"),
//     )
//
//     r.Use(gin.WrapH(csrfMiddleware))
// }

// ========== CSRF 防护清单 ==========
// [ ] 使用 CSRF Token
// [ ] 设置 SameSite=Strict/Lax
// [ ] API 使用自定义头 (X-Requested-With)
// [ ] JSON API 不受 CSRF 影响 (需要正确的 Content-Type)
// [ ] 验证 Origin/Referer
// [ ] 敏感操作需二次确认 (密码/SMS)

// ========== CSRF vs CORS ==========
// CSRF: 利用用户已登录的身份发送恶意请求
// CORS: 控制哪些网站可以访问 API

// CSRF 主要针对 Cookie 认证
// JWT 放在 Authorization 头不受 CSRF 影响
```


## 速率限制 (Rate Limiting)


```
// ========== 限流算法 ==========
// 1. 令牌桶 (Token Bucket)
//    每秒添加固定数量令牌
//    请求需要消耗令牌
//    支持突发流量

// 2. 漏桶 (Leaky Bucket)
//    固定速率处理请求
//    超出桶容量的请求丢弃
//    平滑流量

// 3. 滑动窗口 (Sliding Window)
//    统计最近 N 秒的请求数
//    比固定窗口更准确

// 4. 固定窗口 (Fixed Window)
//    每秒/每分钟计数
//    边界有突发问题

// ========== Go 令牌桶限流 ==========
// 使用 golang.org/x/time/rate

import "golang.org/x/time/rate"

func RateLimitMiddleware(rps int, burst int) gin.HandlerFunc {
    limiter := rate.NewLimiter(rate.Limit(rps), burst)
    return func(c *gin.Context) {
        if !limiter.Allow() {
            c.Header("Retry-After", "1")
            c.AbortWithStatusJSON(429, gin.H{
                "error": "too many requests",
                "retry_after": 1,
            })
            return
        }
        c.Next()
    }
}

// ========== IP 限流 (内存) ==========
type IPRateLimiter struct {
    visitors map[string]*rate.Limiter
    mu       sync.Mutex
    rate     rate.Limit
    burst    int
}

func NewIPRateLimiter(r rate.Limit, b int) *IPRateLimiter {
    return &IPRateLimiter{
        visitors: make(map[string]*rate.Limiter),
        rate:     r,
        burst:    b,
    }
}

func (l *IPRateLimiter) GetLimiter(ip string) *rate.Limiter {
    l.mu.Lock()
    defer l.mu.Unlock()

    limiter, exists := l.visitors[ip]
    if !exists {
        limiter = rate.NewLimiter(l.rate, l.burst)
        l.visitors[ip] = limiter
    }
    return limiter
}

// ========== 限流响应头 ==========
// X-RateLimit-Limit: 100       (上限)
// X-RateLimit-Remaining: 95    (剩余)
// X-RateLimit-Reset: 1640995200 (重置时间)
// Retry-After: 1               (秒)
```


## 安全头与 Helmet


```
// ========== 安全响应头 ==========
// 1. Content-Security-Policy (CSP)
//    控制哪些资源可以加载
//    default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'

// 2. Strict-Transport-Security (HSTS)
//    强制 HTTPS
//    max-age=31536000; includeSubDomains

// 3. X-Frame-Options
//    防止点击劫持 (Clickjacking)
//    DENY / SAMEORIGIN

// 4. X-Content-Type-Options
//    防止 MIME 类型嗅探
//    nosniff

// 5. X-XSS-Protection
//    启用 XSS 过滤 (已废弃, 用 CSP 替代)
//    0 (禁用)

// 6. Referrer-Policy
//    控制 Referer 头发送
//    strict-origin-when-cross-origin

// 7. Permissions-Policy
//    控制浏览器 API 权限
//    camera=(), microphone=(), geolocation=()

// ========== Go 安全头中间件 ==========
func SecurityHeadersMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        c.Header("Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:")
        c.Header("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        c.Header("X-Frame-Options", "DENY")
        c.Header("X-Content-Type-Options", "nosniff")
        c.Header("Referrer-Policy", "strict-origin-when-cross-origin")
        c.Header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        c.Next()
    }
}

// ========== Helmet.js (Node.js) ==========
// Node.js 安全头中间件
// import helmet from 'helmet';
// app.use(helmet());

// Helmet 自动设置:
// Content-Security-Policy
// Cross-Origin-Embedder-Policy
// Cross-Origin-Opener-Policy
// Cross-Origin-Resource-Policy
// Expect-CT
// Origin-Agent-Cluster
// Referrer-Policy
// Strict-Transport-Security
// X-Content-Type-Options
// X-DNS-Prefetch-Control
// X-Download-Options
// X-Frame-Options
// X-Permitted-Cross-Domain-Policies
// X-Powered-By (移除)
// X-XSS-Protection
```


> **Note:** 💡 API 安全要点: CORS 白名单特定 origin, 不用 *; CSRF Token + SameSite Cookie; 限流算法: 令牌桶/漏桶/滑动窗口; 安全头: CSP/HSTS/X-Frame-Options/X-Content-Type-Options; Helmet.js 一站式 Node.js 安全头; X-RateLimit-* 响应头告知客户端; 429 + Retry-After; REST API 用 JWT (Authorization 头) 不受 CSRF 影响。


## 练习


<!-- Converted from: 3_API 安全与防护.html -->
