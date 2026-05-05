# 安全与幂等方法

## 🔒 安全与幂等方法

HTTP 安全方法定义、幂等性深入理解、幂等键实现、重试语义与保障策略。

## 安全方法 (Safe Methods)

安全方法是指不会修改服务器状态的 HTTP 方法——语义上只读。
```
// ========== 安全方法 ==========
// 定义: 调用者不承担修改服务器状态的义务
// 实现: 服务器不应因安全方法而产生副作用
//
// 安全方法列表:
//   GET     — 获取资源
//   HEAD    — 获取响应头 (不返回体)
//   OPTIONS — 查询支持的方法
//   TRACE   — 回显请求
//
// 非安全方法:
//   POST, PUT, PATCH, DELETE

// ========== 安全 ≠ 无副作用 ==========
// 安全方法可以产生副作用,但不应"被期望"产生副作用
//
// 可接受的副作用:
//   GET /articles/123 → 记录访问日志
//   GET /api/search   → 缓存查询结果
//   GET /images/logo  → CDN 缓存
//
// 不可接受的副作用:
//   GET /delete/user/123 → 删除用户! ❌
//   GET /api/checkout   → 创建订单! ❌
//
// ========== 实际意义 ==========
// 安全方法可被:
//   - 浏览器安全地预加载 (prefetch/preload)
//   - CDN 安全地缓存
//   - 爬虫安全地访问
//   - 链接安全地分享
```
## 幂等性 (Idempotency)

幂等指多次执行同一操作的结果与执行一次相同。
```
// ========== 数学定义 ==========
// f(f(x)) = f(x)
// 重复应用函数不会改变结果
//
// 数学幂等例子:
//   abs(abs(-5)) = abs(-5) = 5
//   max(5, max(5, 3)) = max(5, 3) = 5
//
// ========== HTTP 幂等 ==========
// 自然幂等方法:
//   GET:   多次读取同一资源,结果相同
//   PUT:   多次 PUT 相同数据,资源不变
//   DELETE:多次删除同个资源,结果相同 (404 也是相同)
//   HEAD:  同 GET,不返回体
//   OPTIONS:查询总是返回相同结果
//
// 非幂等方法:
//   POST:  多次提交创建多个资源
//   PATCH:  增量操作可能不幂等
//
// ========== DELETE 幂等示例 ==========
// 第 1 次: DELETE /users/123 → 200 OK (删除成功)
// 第 2 次: DELETE /users/123 → 404 Not Found (已删除)
// 第 3 次: DELETE /users/123 → 404 Not Found (已删除)
//
// 服务器状态: 都是"用户 123 不存在"
// → 幂等成立
```
## 幂等键 (Idempotency-Key)

对于天然非幂等的 POST/PATCH，用幂等键保证安全重试。
```
// ========== 幂等键机制 ==========
// 客户端在请求头中提供唯一键:
//   POST /api/payments
//   Idempotency-Key: 550e8400-e29b-41d4-a716-446655440000
//   { "amount": 100, "currency": "USD" }
//
// 服务器:
//   1. 检查该键是否已处理 → 返回之前的结果
//   2. 未处理 → 执行并缓存结果
//   3. 返回结果
//
// 流程:
//   客户端 → POST /payments (key=K1) → 服务器处理 → 返回 200
//   客户端 → POST /payments (key=K1) → 返回缓存 200 (重复!)
//   客户端 → POST /payments (key=K2) → 新处理 → 返回 200

// ========== 幂等键实现 ==========
// Node.js 中间件示例:
//
//   const idempotency = new Map(); // 或用 Redis
//
//   app.post('/payments', async (req, res) => {
//     const key = req.headers['idempotency-key'];
//     if (!key) return res.status(400).send('Missing key');
//
//     if (idempotency.has(key)) {
//       return res.json(idempotency.get(key));
//     }
//
//     const result = await processPayment(req.body);
//     idempotency.set(key, result);
//     // 设置 TTL 防止内存泄漏
//     setTimeout(() => idempotency.delete(key), 86400000);
//     res.json(result);
//   });
```
> **Note**: 💳 幂等键在支付场景中至关重要——你的银行不会因为网络重试而重复扣款。Stripe 的幂等键是业界标准实现。

## 重试语义与最佳实践
```
// ========== 重试策略 ==========
// 客户端网络请求失败时自动重试
// 需要区分可重试和不可重试:
//
// 可自动重试:
//   网络超时 (ETIMEDOUT) → 幂等方法安全,非幂等方法需幂等键
//   服务器 5xx → 可能重试 (服务器未处理)
//   HTTP 429 Too Many Requests → 带 Retry-After 重试
//
// 不可自动重试:
//   客户端 4xx (除 429) → 请求本身有问题
//   POST 且无幂等键 → 不确定服务器是否已处理

// ========== 幂等键最佳实践 ==========
// 1. 幂等键必须是全局唯一的
//    推荐使用 UUID v4
//   也可用哈希 (用户ID + 时间戳 + 随机数)
//
// 2. 键值存储需设置 TTL
//   至少 24 小时
//   支付场景建议 7 天
//
// 3. 键到期后的重试风险
//   可接受的业务风险
//   关键操作可人工审核

// ========== 条件请求策略 ==========
// 用 ETag/If-Match 实现条件更新:
//
//   GET /users/123 → ETag: "v1"
//   PUT /users/123
//   If-Match: "v1"  ← 只有版本匹配才更新
//   { name: "Alice" }
//
// 如果中途被别人改了,服务器返回 412 Precondition Failed
// 客户端需要重新获取最新版本再决定如何处理
```
