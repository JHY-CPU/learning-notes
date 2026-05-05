# 请求与响应结构总结

## 📝 请求与响应结构总结

HTTP 报文结构综合回顾、curl 实践、浏览器 DevTools 网络面板解读。

## HTTP 报文结构回顾
```
// ========== 请求报文 ==========
// 请求行:  方法 URL 版本
// 请求头:  Key: Value
// 空行:    \r\n
// 请求体:  (可选)
//
// POST /api/users HTTP/1.1
// Host: api.example.com
// Content-Type: application/json
// Authorization: Bearer
//
// {"name": "Alice", "email": "alice@a.com"}

// ========== 响应报文 ==========
// 状态行:  版本 状态码 原因短语
// 响应头:  Key: Value
// 空行:    \r\n
// 响应体:  (可选)
//
// HTTP/1.1 201 Created
// Location: /api/users/123
// Content-Type: application/json
//
// {"id": 123, "name": "Alice"}

// ========== 重要规则 ==========
// 1. 每行以 \r\n (CRLF) 结束
// 2. 头部与体之间有一个空行 (\r\n)
// 3. Content-Length 指示体长度 (字节)
// 4. 或 Transfer-Encoding: chunked 逐块传输
// 5. 头部名不区分大小写 (Content-Type = content-type)
// 6. 头部名后跟 : 和空格
// 7. 请求行/状态行是单行
```
## 使用 curl 查看 HTTP 报文
```
// ========== curl 命令 ==========
//
// 基本 GET 请求:
//   $ curl https://api.example.com/users
//
// 显示完整请求/响应头:
//   $ curl -v https://api.example.com/users
//   > GET /users HTTP/1.1
//   > Host: api.example.com
//   > User-Agent: curl/8.0
//   >
//   < HTTP/1.1 200 OK
//   < Content-Type: application/json
//   <
//   [响应体]
//
// 仅显示响应头:
//   $ curl -I https://api.example.com/users
//
// POST 请求:
//   $ curl -X POST https://api.example.com/users \
//     -H "Content-Type: application/json" \
//     -d '{"name":"Alice"}'
//
// 自定义请求头:
//   $ curl -H "Authorization: Bearer " \
//     -H "X-Custom-Header: value" \
//     https://api.example.com/users
//
// 跟踪重定向:
//   $ curl -L http://example.com
//
// 设置超时:
//   $ curl --connect-timeout 5 --max-time 30 https://example.com
```
## 浏览器 DevTools Network 面板
```
// ========== Network 面板关键信息 ==========
//
// 请求列表显示:
//   Name:      请求的资源名
//   Status:    HTTP 状态码
//   Type:      资源类型 (document/script/style)
//   Initiator: 发起者 (解析器/脚本/其他)
//   Size:      资源大小 (传输大小 / 实际大小)
//              如: 12 KB / 48 KB (gzip 压缩比)
//   Time:      总耗时
//   Waterfall: 时间线瀑布图
//
// ========== 瀑布图各阶段 ==========
// Queueing:          等待队列 (浏览器限制并发数)
// DNS Lookup:        DNS 解析
// Initial connection: TCP 三次握手
// SSL/TLS:           TLS 握手
// Request sent:      发送请求
// Waiting (TTFB):    等待服务器响应首字节
// Content Download:  下载响应内容
//
// ========== 优化指标 ==========
// TTFB (Time To First Byte):
//   理想 < 200ms, 良好 < 500ms
//   取决于: 服务器处理 + 网络延迟
//
// 内容下载时间:
//   取决于: 文件大小 + 带宽
//   优化: 压缩, CDN, 缓存

// ========== 常见 HTTP 版本判断 ==========
// HTTP/1.1: 请求列表显示 Connection ID
// HTTP/2:   显示同一连接的多个并行流
// HTTP/3:   显示 "h3" 协议标识
// 通过 Protocol 列查看 (右键添加列)
```
> **Note**: 🔍 打开 Chrome DevTools → Network 面板,刷新页面看瀑布图——DNS 查询花了多久? TLS 握手多长? TTFB 是否合理? 这是每个后端开发者都应该会的基本诊断技能。

## HTTP 调试最佳实践
```
// ========== HTTP 调试工具 ==========
// curl:              命令行调试,脚本化
// httpie:            更友好的 curl 替代
// Postman:           GUI 工具,集合管理
// Insomnia:          GraphQL + REST 客户端
// Chrome DevTools:   浏览器网络抓包
// Wireshark:         网络级抓包分析
// Charles/Fiddler:   代理抓包 (HTTPS)

// ========== 状态码速查 ==========
// 200:  请求成功
// 201:  创建成功
// 204:  删除成功
// 301:  永久重定向 (SEO)
// 302:  临时重定向
// 304:  缓存未过期
// 400:  请求格式错
// 401:  未认证
// 403:  无权限
// 404:  未找到
// 409:  资源冲突
// 422:  验证失败
// 429:  限流
// 500:  服务器内部错误
// 502:  网关错误
// 503:  服务不可用
// 504:  网关超时

// ========== 常见故障排查 ==========
// 问题: 发送 POST 请求收到 404
//   检查: URL 路径是否正确? 方法是否支持?
//
// 问题: 跨域请求被阻止
//   检查: CORS 头是否配置? OPTIONS 预检?
//
// 问题: API 响应速度慢
//   检查: TTFB 是否高? 数据库查询? 缓存?
```
