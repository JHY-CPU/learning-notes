# CORS深入


## CORS 深入


简单/复杂请求区分、预检请求缓存、withCredentials、通配符限制。


## CORS API


```
// ========== 简单请求 ==========
// 方法: GET / HEAD / POST
// Content-Type: application/x-www-form-urlencoded
//              multipart/form-data
//              text/plain
// 无自定义头

// ========== 复杂请求 ==========
// 方法: PUT / DELETE / PATCH
// Content-Type: application/json
// 自定义头
// → 先发 OPTIONS 预检

// ========== 服务端 CORS 头 ==========
Access-Control-Allow-Origin: https://example.com
Access-Control-Allow-Methods: GET, POST, PUT
Access-Control-Allow-Headers: Content-Type, Authorization
Access-Control-Max-Age: 86400
Access-Control-Allow-Credentials: true
Access-Control-Expose-Headers: X-Total-Count
```


## 演示：CORS

点击按钮查看


<!-- Converted from: 64_CORS深入.html -->
