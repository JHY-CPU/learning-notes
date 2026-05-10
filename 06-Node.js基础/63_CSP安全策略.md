# CSP安全策略


## CSP 安全策略


Content-Security-Policy 指令、script-src/style-src/img-src、report-only。


## CSP API


```
// ========== HTTP 头 ==========
Content-Security-Policy: default-src 'self';
    script-src 'self' https://cdn.example.com;
    style-src 'self' 'unsafe-inline';
    img-src 'self' data: https:;
    connect-src 'self' https://api.example.com;
    font-src 'self' https://fonts.gstatic.com;
    frame-src 'none';
    object-src 'none';

// ========== meta 标签 ==========


// ========== 报告模式 ==========
Content-Security-Policy-Report-Only: default-src 'self';
    report-uri /csp-report;
```


## 演示：CSP

点击按钮查看


<!-- Converted from: 63_CSP安全策略.html -->
