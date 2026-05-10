# Express OAuth 与刷新令牌


## 🔄 Express OAuth 与刷新令牌


Refresh Token 旋转机制、OAuth 2.0 授权码流程 (Authorization Code + PKCE)、令牌存储策略 (localStorage vs Cookie)、令牌自动刷新 (Axios 拦截器)、多设备令牌管理、安全退出所有设备。


## Refresh Token 旋转


```
// ========== Refresh Token 旋转 ==========
// 每次使用 refresh token 就换一个新的
// 防止 refresh token 泄露后的持续访问

// utils/tokenService.js:
const crypto = require('crypto');
const jwt = require('jsonwebtoken');

class TokenService {
    constructor() {
        this.refreshTokenExpiry = 7 * 24 * 60 * 60; // 7 天 (秒)
    }

    async generateTokens(user) {
        // Access Token
        const accessToken = jwt.sign(
            { sub: user._id, role: user.role },
            process.env.JWT_SECRET,
            { expiresIn: '15m' }
        );

        // Refresh Token (随机字符串, 存 DB/Redis)
        const refreshToken = crypto.randomBytes(40).toString('hex');
        const expiresAt = new Date(Date.now() + this.refreshTokenExpiry * 1000);

        // 存到数据库 (可多设备)
        await RefreshToken.create({
            user: user._id,
            token: refreshToken,
            expiresAt,
            family: crypto.randomBytes(8).toString('hex'), // 令牌族
        });

        return { accessToken, refreshToken, expiresAt };
    }

    async rotateRefreshToken(oldToken, user) {
        // 1. 查找当前 refresh token
        const stored = await RefreshToken.findOne({ token: oldToken });

        if (!stored) {
            // 令牌不存在 → 可能有攻击, 使该族所有令牌失效
            return null;
        }

        // 2. 删除旧令牌
        await RefreshToken.deleteOne({ _id: stored._id });

        // 3. 生成新令牌 (同族)
        const newToken = crypto.randomBytes(40).toString('hex');
        const expiresAt = new Date(Date.now() + this.refreshTokenExpiry * 1000);

        await RefreshToken.create({
            user: user._id,
            token: newToken,
            expiresAt,
            family: stored.family,
        });

        return { refreshToken: newToken, expiresAt };
    }

    async revokeAllUserTokens(userId) {
        // 退出所有设备
        await RefreshToken.deleteMany({ user: userId });
    }
}

module.exports = new TokenService();
```


## 令牌存储策略


```
// ========== 令牌存储 ==========
// ┌────────────────┬────────────────────────┬────────────────┐
// │ 方案            │ 存储位置              │ 风险           │
// ├────────────────┼────────────────────────┼────────────────┤
// │ localStorage   │ 浏览器                 │ XSS 可读取     │
// │ httpOnly Cookie│ Cookie                 │ 防 XSS, CSRF  │
// │ 内存 + 刷新    │ JS 变量 + Cookie       │ 安全, 复杂     │
// └────────────────┴────────────────────────┴────────────────┘

// ========== 推荐: httpOnly Cookie ==========
// Access Token 和 Refresh Token 都存 httpOnly Cookie
// 防 XSS 窃取, 自动发送

// 设置 Cookie:
function setTokenCookies(res, tokens) {
    // Access Token (短过期)
    res.cookie('accessToken', tokens.accessToken, {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        sameSite: 'strict',
        maxAge: 15 * 60 * 1000,          // 15 分钟
        path: '/',
    });

    // Refresh Token
    res.cookie('refreshToken', tokens.refreshToken, {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        sameSite: 'strict',
        maxAge: 7 * 24 * 60 * 60 * 1000, // 7 天
        path: '/api/auth',               // 只在 /api/auth 路径发送
    });
}

// ========== Axios 拦截器 (前端) ==========
// 自动刷新令牌:
axios.interceptors.response.use(
    (response) => response,
    async (error) => {
        const originalRequest = error.config;

        if (error.response?.status === 401 && !originalRequest._retry) {
            originalRequest._retry = true;

            try {
                // 调用刷新端点 (Cookie 自动携带 refreshToken)
                const { data } = await axios.post('/api/auth/refresh');
                // 新 Cookie 自动设置
                return axios(originalRequest);
            } catch (refreshError) {
                // 刷新失败 → 跳转登录
                window.location.href = '/login';
                return Promise.reject(refreshError);
            }
        }

        return Promise.reject(error);
    }
);
```


## OAuth 2.0 授权码流程


```
// ========== OAuth 2.0 授权码模式 ==========
// 适合第三方应用接入

// ┌────────┐    ┌──────────┐    ┌───────────┐
// │ 用户   │    │ 应用      │    │ 授权服务器│
// └───┬────┘    └────┬─────┘    └─────┬─────┘
//     │ 1. 请求授权  │                │
//     │─────────────>│                │
//     │ 2. 重定向    │                │
//     │<─────────────│                │
//     │ 3. 用户同意  │                │
//     │─────────────────────────────>│
//     │ 4. 授权码    │                │
//     │<─────────────────────────────│
//     │ 5. 授权码    │                │
//     │─────────────>│                │
//     │ 6. 换令牌    │                │
//     │              │────────────────>│
//     │ 7. access/refresh             │
//     │              │<────────────────│
//     │ 8. 完成      │                │
//     │<─────────────│                │
// └───┴────┘    └────┴─────┘    └─────┴─────┘

// ========== PKCE 扩展 ==========
// 公共客户端 (SPA/移动端) 使用 PKCE
// code_verifier + code_challenge (S256)

// 1. 生成 code_verifier (随机字符串)
const verifier = crypto.randomBytes(32).toString('base64url');

// 2. code_challenge = SHA256(verifier)
const challenge = crypto
    .createHash('sha256')
    .update(verifier)
    .digest('base64url');

// 3. 授权请求带 code_challenge
const authUrl = `https://provider.com/auth?
    response_type=code&
    client_id=xxx&
    redirect_uri=xxx&
    code_challenge=${challenge}&
    code_challenge_method=S256`;

// 4. 令牌请求带 code_verifier
fetch('https://provider.com/token', {
    method: 'POST',
    body: JSON.stringify({
        grant_type: 'authorization_code',
        code: 'xxx',
        code_verifier: verifier,  // 验证
        redirect_uri: 'xxx',
    }),
});
```


## 多设备令牌管理


```
// ========== 多设备管理 ==========
// RefreshToken 模型:

const refreshTokenSchema = new mongoose.Schema({
    user: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true,
    },
    token: {
        type: String,
        required: true,
        index: true,
    },
    family: String,            // 令牌族 (用于检测重放)
    deviceInfo: {
        name: String,          // "Chrome on Windows"
        ip: String,
        userAgent: String,
    },
    expiresAt: {
        type: Date,
        required: true,
        index: { expireAfterSeconds: 0 },  // TTL 自动清理
    },
    lastUsedAt: Date,
    isRevoked: { type: Boolean, default: false },
}, { timestamps: true });

// ========== 设备管理 API ==========
// 列出所有活跃设备:
app.get('/auth/devices', authenticate, asyncHandler(async (req, res) => {
    const devices = await RefreshToken.find({
        user: req.user._id,
        isRevoked: false,
        expiresAt: { $gt: new Date() },
    }).select('deviceInfo lastUsedAt createdAt');

    res.success(devices);
}));

// 退出特定设备:
app.delete('/auth/devices/:id', authenticate, asyncHandler(async (req, res) => {
    await RefreshToken.findOneAndUpdate(
        { _id: req.params.id, user: req.user._id },
        { isRevoked: true }
    );
    res.success(null, 'Device logged out');
}));

// 退出所有设备:
app.post('/auth/logout-all', authenticate, asyncHandler(async (req, res) => {
    await RefreshToken.updateMany(
        { user: req.user._id },
        { isRevoked: true }
    );
    res.success(null, 'All devices logged out');
}));
```


> **Note:** 💡 Refresh Token 要点: 短过期 access + 长过期 refresh; 旋转机制 (每次刷新换新); httpOnly Cookie 存令牌最安全 (防 XSS); PKCE 保护公共客户端; 多设备管理用令牌族检测重放; TTL 索引自动清理过期; 退出所有设备 = 删除所有 refresh token; 密码修改后应撤销所有令牌。


## 练习


<!-- Converted from: 22_Express OAuth 与刷新令牌.html -->
