# GraphQL 与 REST 对比

## 一、请求对比

```
REST - 多次请求获取用户 + 订单 + 商品:
GET /api/users/1
GET /api/users/1/orders
GET /api/orders/O001/items

GraphQL - 一次请求获取所有数据:
query {
  user(id: "1") {
    name
    orders {
      id
      items {
        product { name price }
        quantity
      }
    }
  }
}
```

## 二、特性对比

| 特性 | REST | GraphQL |
|------|------|---------|
| 端点数量 | 多个 | 单一 |
| 数据获取 | 固定结构 | 按需获取 |
| 过度/不足获取 | 常见 | 不存在 |
| 版本管理 | URL 版本 | Schema 演进 |
| 缓存 | HTTP 缓存简单 | 需客户端缓存 |
| 文件上传 | 原生支持 | 需额外处理 |
| 实时通信 | WebSocket | Subscription |
| 学习曲线 | 低 | 中等 |
| 错误处理 | HTTP 状态码 | 响应体 errors |
| 工具生态 | 成熟 | 快速发展 |

## 三、适用场景

```yaml
适合 REST:
  - 简单 CRUD 操作
  - 文件上传/下载
  - 缓存敏感场景
  - 公开 API
  - 微服务间内部调用

适合 GraphQL:
  - 复杂数据关系
  - 多端应用 (Web/iOS/Android)
  - 需要灵活查询
  - 聚合多个数据源
  - 移动端减少网络请求
```

## 四、混合使用

```
架构示例:
┌─────────────┐
│   客户端     │
└──────┬──────┘
       │ GraphQL
┌──────┴──────┐
│  BFF / API  │ ← GraphQL 聚合层
│   Gateway   │
└──┬───┬───┬──┘
   │   │   │  REST/gRPC
   ▼   ▼   ▼
  用户 订单 商品 ← 内部微服务
```

## 五、缓存对比

```
REST 缓存:
┌───────────┐     ┌───────────┐     ┌───────────┐
│  客户端    │────▶│ CDN/代理   │────▶│  服务端    │
│ Cache-Control │  │ ETag/304  │     │ 响应缓存  │
└───────────┘     └───────────┘     └───────────┘
- HTTP 缓存天然支持
- URL 即缓存键
- CDN 友好

GraphQL 缓存:
┌───────────┐     ┌───────────────────────┐
│  客户端    │────▶│ Apollo InMemoryCache  │
│ 按 ID 缓存 │    │ normalized cache      │
└───────────┘     └───────────────────────┘
- 所有请求同一端点，无法用 HTTP 缓存
- 需客户端按实体 ID 归一化缓存
- APQ (Automatic Persisted Queries) 可启用 CDN
```

## 六、错误处理对比

```yaml
REST 错误处理:
  方式: HTTP 状态码 + 响应体
  200: 成功
  400: 请求错误
  401: 未认证
  403: 无权限
  404: 资源不存在
  500: 服务器错误
  优点: 浏览器/工具天然支持
  缺点: 混合成功/失败难表达

GraphQL 错误处理:
  方式: 200 状态码 + errors 字段
  响应:
    {
      "data": { "user": null },
      "errors": [{
        "message": "未认证",
        "path": ["user"],
        "extensions": { "code": "UNAUTHENTICATED" }
      }]
    }
  优点: 精确到字段的错误
  缺点: 所有请求都返回 200
```

## 七、迁移策略

```yaml
从 REST 迁移到 GraphQL:
  阶段 1 - 包装:
    - 在 REST 服务前加 GraphQL 层
    - Resolver 直接调用 REST API
    - 不改后端逻辑

  阶段 2 - 重构:
    - 逐步将业务逻辑移入 Resolver
    - REST API 保留用于内部调用
    - 客户端逐步切换

  阶段 3 - 优化:
    - DataLoader 优化数据获取
    - 客户端缓存策略
    - REST API 可以降级为内部 API
```

## 八、注意事项

1. **不是非此即彼**，可以混合使用
2. **GraphQL 适合 BFF 聚合层**
3. **内部微服务间 REST/gRPC 更简单**
4. **缓存策略是 GraphQL 的弱项**
5. **根据团队和项目需求选择**
6. **迁移要渐进式**，从包装现有 REST 开始
7. **错误处理要统一规范**，使用 extensions.code
