# XSS跨站脚本


## XSS 跨站脚本


反射型/存储型/DOM 型 XSS、转义策略、CSP 防御、XSS 检测。


## XSS API


```
// ========== 反射型 XSS ==========
// URL 参数直接注入到页面
// https://site.com/search?q=

// ========== 存储型 XSS ==========
// 恶意代码存入数据库，其他用户访问时执行
// 评论/留言/用户资料等

// ========== DOM 型 XSS ==========
// 客户端 JS 将用户输入插入 DOM
element.innerHTML = userInput;  // ❌

// ========== 防御 ==========
// 1. 输出转义 (HTML Entity)
function escapeHTML(str) {
    return str.replace(/&/g, '&')
              .replace(//g, '>')
              .replace(/"/g, '"')
              .replace(/'/g, ''');
}
// 2. 使用 textContent 代替 innerHTML
// 3. CSP 策略
// 4. 输入验证
```


## 演示：XSS

点击按钮查看


## 什么是 XSS

XSS（Cross-Site Scripting，跨站脚本攻击）是指攻击者将恶意脚本注入到网页中，在用户浏览器执行。可窃取Cookie、会话Token、修改页面内容等。

## 三种类型详解

1. **反射型XSS**：恶意代码通过URL参数传入，服务器反射回页面。需要诱导用户点击构造好的链接。
2. **存储型XSS**：恶意代码存入数据库（如评论区），所有访问者都会中招。危害最大。
3. **DOM型XSS**：前端JS将用户输入不安全地插入DOM（如 `innerHTML`），不经过服务端。

## 防御手段

- **输出编码**：HTML转义 `< > & " '` 等特殊字符
- **使用安全API**：`textContent` 代替 `innerHTML`，`setAttribute` 代替字符串拼接
- **CSP策略**：限制脚本来源，禁止inline脚本
- **输入验证**：对用户输入做白名单校验
- **Cookie防护**：`HttpOnly` 阻止JS读取Cookie
- **模板引擎**：使用自动转义的模板引擎

## Node.js 后端防御

在Express/Koa中使用 `helmet` 中间件自动设置安全头，使用模板引擎的自动转义功能，对用户输入做净化处理（如 `xss` 或 `DOMPurify` 库）。

<!-- Converted from: 61_XSS跨站脚本.html -->
