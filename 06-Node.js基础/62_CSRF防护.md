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


<!-- Converted from: 62_CSRF防护.html -->
