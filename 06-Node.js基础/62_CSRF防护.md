# CSRF防护


## CSRF 防护


CSRF 原理、SameSite Cookie、Token 验证(CSRF Token)、Referer 头。


## CSRF API


```
// ========== CSRF 攻击流程 ==========
// 用户已登录 bank.com (有 Cookie)
// 用户访问恶意网站 attacker.com
// 恶意网站发起请求 bank.com/transfer?to=attacker
// Cookie 自动携带 → 请求通过

// ========== 防御方案 ==========
// 1. SameSite Cookie (Lax/Strict)
// 2. CSRF Token (双重提交)
// 3. Referer/Origin 验证
// 4. 自定义 Header (X-Requested-With)
// 5. 验证码 / 二次确认
```


## 演示：CSRF

点击按钮查看


## 什么是 CSRF

CSRF（Cross-Site Request Forgery，跨站请求伪造）是指攻击者利用用户已登录的身份，在用户不知情的情况下向目标网站发起恶意请求。浏览器会自动携带目标网站的Cookie，导致请求被服务器信任。

## 攻击原理

用户登录A网站后，浏览器保存了A网站的Cookie。用户访问恶意B网站时，B网站通过 `<img>`、`<form>`、`fetch` 等向A网站发起请求，浏览器自动携带A网站的Cookie，A网站误认为是用户本人操作。

## 防御方案详解

1. **SameSite Cookie**：`Strict` 完全禁止跨站携带Cookie，`Lax` 允许GET导航跳转，推荐 `Lax`
2. **CSRF Token**：服务端生成Token嵌入表单，提交时验证Token是否匹配
3. **双重Cookie验证**：服务端要求请求同时携带Cookie和自定义Header中的Token
4. **Origin/Referer验证**：检查请求来源是否合法
5. **自定义Header**：AJAX请求添加自定义头，简单请求不会被跨域发送

## Node.js 实现

使用 `csurf` 中间件生成和验证CSRF Token。Express中 `app.use(csurf({ cookie: true }))` 即可启用。前端在表单中加入隐藏字段或在AJAX请求头中携带Token。

## Token 实现与前端集成

```javascript
// ========== Node.js CSRF Token 生成 ==========
const crypto = require('crypto');

function generateCsrfToken() {
    return crypto.randomBytes(32).toString('hex');
}

// ========== Express 中间件 ==========
function csrfProtection(req, res, next) {
    if (['GET', 'HEAD', 'OPTIONS'].includes(req.method)) {
        // 生成 Token 并存入 session
        req.session.csrfToken = generateCsrfToken();
        res.locals.csrfToken = req.session.csrfToken;
        return next();
    }

    // 验证 POST/PUT/DELETE 请求
    const token = req.body._csrf || req.headers['x-csrf-token'];
    if (!token || token !== req.session.csrfToken) {
        return res.status(403).json({ error: 'Invalid CSRF token' });
    }
    next();
}

// ========== 前端集成 ==========
// 方式1: 表单隐藏字段
// <form method="POST">
//     <input type="hidden" name="_csrf" value="<%= csrfToken %>">
//     <button type="submit">提交</button>
// </form>

// 方式2: AJAX Header
const csrfToken = document.querySelector('meta[name="csrf-token"]')?.content;
fetch('/api/data', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'X-CSRF-Token': csrfToken,
    },
    body: JSON.stringify({ data: 'value' }),
});

// ========== Axios 拦截器 ==========
axios.interceptors.request.use(config => {
    const token = document.querySelector('meta[name="csrf-token"]')?.content;
    if (token) config.headers['X-CSRF-Token'] = token;
    return config;
});

// ========== SameSite Cookie 最佳实践 ==========
// Set-Cookie: session=abc; SameSite=Lax; Secure; HttpOnly
// Lax: 允许顶层GET导航跳转，禁止跨站 POST/iframe/img 携带 Cookie
// Strict: 完全禁止跨站携带 (用户体验差)
// None: 允许跨站 (必须配合 Secure，仅 HTTPS)

// ========== Node.js 设置 ==========
const express = require('express');
const cookieParser = require('cookie-parser');
app.use(cookieParser());
app.use((req, res, next) => {
    res.cookie('session', token, {
        httpOnly: true,
        secure: true,
        sameSite: 'lax',
        maxAge: 24 * 60 * 60 * 1000,
    });
    next();
});
```

## 完整防护策略

```javascript
// ========== 多层防护方案 ==========
// 1. SameSite Cookie — 第一道防线
// 2. CSRF Token — 核心验证
// 3. Origin/Referer 检查 — 辅助验证
// 4. 自定义 Header — AJAX 必须

function multiLayerCsrf(req, res, next) {
    // 1. Origin 检查
    const origin = req.headers.origin;
    const allowed = ['https://myapp.com', 'https://www.myapp.com'];
    if (origin && !allowed.includes(origin)) {
        return res.status(403).json({ error: 'Invalid origin' });
    }

    // 2. 自定义 Header 检查 (简单请求不会自定义头)
    if (['POST', 'PUT', 'DELETE'].includes(req.method)) {
        if (!req.headers['x-requested-with']) {
            return res.status(403).json({ error: 'Missing custom header' });
        }
    }

    // 3. Token 验证 (上面已实现)

    next();
}
```

<!-- Converted from: 62_CSRF防护.html -->
