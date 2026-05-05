# 缓存头Cache-Control与ETag

## 💾 缓存头 Cache-Control 与 ETag

Cache-Control 指令详解、强缓存与协商缓存、ETag/If-None-Match、缓存策略最佳实践。

## 缓存概述

HTTP 缓存可以显著提升性能，减少带宽消耗，降低服务器负载。缓存分为强缓存和协商缓存。
```
// ========== 缓存类型 ==========
//
// 强缓存 (本地缓存):
//   浏览器直接从本地缓存读取,不发请求
//   状态: 200 (from disk/memory cache)
//   速度: 最快 (0ms)
//
// 协商缓存 (条件请求):
//   发送请求询问服务器资源是否有变化
//   无变化 → 304 Not Modified (用缓存)
//   有变化 → 200 + 新资源
//   速度: 需一次网络往返
//
// ========== 缓存流程 ==========
// 1. 浏览器请求资源
// 2. 检查强缓存 → 未过期 → 直接用缓存
// 3. 强缓存过期 → 协商缓存 → 带条件头请求
// 4. 服务器判断资源未变 → 304 → 用缓存
// 5. 资源已变 → 200 + 新资源 → 更新缓存
```
## Cache-Control 指令
```
// ========== Cache-Control 指令 ==========
//
// max-age=:      缓存有效期 (相对时间)
// s-maxage=:     共享缓存有效期 (CDN/代理)
// public:                 可被任何缓存存储
// private:                仅浏览器可缓存
// no-cache:               每次需验证 (协商缓存)
// no-store:               完全禁止缓存
// must-revalidate:        过期后必须验证
// proxy-revalidate:       代理缓存过期后验证
// immutable:              资源永远不会变 (304 都不需要)
// stale-while-revalidate: 过期后可用旧缓存,后台更新
// stale-if-error:         服务器错误时可用过期缓存

// ========== 策略示例 ==========
// 静态资源 (不变的文件):
//   Cache-Control: public, max-age=31536000, immutable
//   缓存 1 年, 永不变化
//
// HTML 页面 (动态内容):
//   Cache-Control: no-cache
//   每次需验证 (但可能返回 304)
//
// API 响应 (用户数据):
//   Cache-Control: private, max-age=60
//   仅浏览器缓存 1 分钟
//
// 敏感数据 (银行余额):
//   Cache-Control: no-store
//   禁止缓存

// ========== 过期优先顺序 ==========
// 1. Cache-Control: max-age (最高优先级)
// 2. Expires (HTTP/1.0 兼容)
// 3. Last-Modified (启发式缓存)
//
// 没有缓存头时, 浏览器用启发式算法:
//   缓存时间 = (Date - Last-Modified) × 0.1
```
## ETag 与协商缓存
```
// ========== ETag (实体标签) ==========
// 资源版本的唯一标识符 (通常是哈希)
// 资源变化 → ETag 变化
//
// 服务器响应:
//   ETag: "abc123"           (强验证器)
//   ETag: W/"abc123"         (弱验证器,允许微小差异)
//
// 客户端后续请求:
//   If-None-Match: "abc123"
//   服务器比较 ETag:
//     相同 → 304 Not Modified (无响应体)
//     不同 → 200 + 新资源

// ========== Last-Modified / If-Modified-Since ==========
// 服务器响应:
//   Last-Modified: Mon, 01 Jan 2024 12:00:00 GMT
//
// 客户端后续请求:
//   If-Modified-Since: Mon, 01 Jan 2024 12:00:00 GMT
//
// ETag vs Last-Modified:
//   ETag 精确 (内容哈希), 优先使用
//   Last-Modified 只能到秒级
//   两者可同时使用,ETag 优先级更高

// ========== 304 响应 ==========
// 服务器判断资源未变化 → 304 Not Modified
// 响应特点:
//   无响应体 (空的)
//   保留 Last-Modified 或 ETag
//   浏览器用本地缓存的资源
//   比完整请求快很多
```
## 缓存策略最佳实践
```
// ========== 不同资源策略 ==========
//
// 1. URL 含指纹的静态资源 (打包工具自动生成):
//    app.a3b4c5.js
//    style.d6e7f8.css
//    logo.v1.png
//    策略: Cache-Control: public, max-age=31536000, immutable
//    缓存 1 年,下次用新 URL 加载新版本
//
// 2. HTML 页面:
//    index.html
//    策略: Cache-Control: no-cache
//    每次验证,但静态子资源可以长期缓存
//
// 3. API 数据:
//    GET /api/users
//    策略: Cache-Control: private, max-age=60
//    短暂缓存,避免重复请求
//
// 4. 用户头像/图片:
//    /uploads/avatar.jpg
//    策略: Cache-Control: public, max-age=86400
//    缓存 1 天

// ========== CDN 缓存策略 ==========
// 客户端 ← CDN ← 源服务器
//
// s-maxage 控制 CDN 缓存:
//   Cache-Control: public, max-age=60, s-maxage=3600
//   浏览器: 60 秒
//   CDN:    3600 秒
//
// CDN 缓存失效:
//   主动刷新 (API 调用)
//   等待 TTL 到期
//   版本化 URL (推荐)

// ========== 缓存陷阱 ==========
// ❌ 不要缓存敏感数据 (no-store)
// ❌ API 不要用 max-age=31536000 (除非你知道在做什么)
// ❌ no-cache 不是 no-store (no-cache 仍可缓存,只是需要验证)
// ✅ 静态资源用指纹 URL + 长 max-age
// ✅ API 用短 max-age 或 no-cache
// ✅ 使用 ETag 减少带宽
```
> **Note**: ⚡ 缓存是前端性能优化最重要的手段之一。正确配置缓存可以让页面加载时间从秒级降到毫秒级——95% 的静态资源请求可以通过缓存避免。
