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


<!-- Converted from: 61_XSS跨站脚本.html -->
