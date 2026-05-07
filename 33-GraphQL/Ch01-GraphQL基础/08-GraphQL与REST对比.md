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

## 五、注意事项

1. **不是非此即彼**，可以混合使用
2. **GraphQL 适合 BFF 聚合层**
3. **内部微服务间 REST/gRPC 更简单**
4. **缓存策略是 GraphQL 的弱项**
5. **根据团队和项目需求选择**
