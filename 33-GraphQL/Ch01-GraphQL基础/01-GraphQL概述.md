# GraphQL 概述

## 一、核心概念

```
GraphQL 是一种 API 查询语言和运行时:
├── 按需查询 - 客户端决定返回哪些字段
├── 强类型 - Schema 定义所有类型
├── 单一端点 - 所有请求一个 URL
└── 自省 - API 可自我描述
```

## 二、GraphQL vs REST

```yaml
REST:
  - 多个端点: /users, /orders, /products
  - 固定返回结构
  - 可能过度获取或不足获取数据
  - 版本管理: /api/v1/users

GraphQL:
  - 单一端点: /graphql
  - 客户端指定返回字段
  - 精确获取所需数据
  - 无需版本管理，演进式 Schema
```

## 三、基本示例

```graphql
# 查询 - 获取用户及其订单
query {
  user(id: "1") {
    name
    email
    orders {
      id
      total
      status
    }
  }
}

# 返回结果 - 只有所请求的字段
{
  "data": {
    "user": {
      "name": "张三",
      "email": "zhangsan@example.com",
      "orders": [
        { "id": "O001", "total": 299.00, "status": "PAID" }
      ]
    }
  }
}
```

## 四、优势与适用场景

| 优势 | 说明 |
|------|------|
| 减少网络请求 | 一次查询获取关联数据 |
| 避免过度获取 | 只返回请求的字段 |
| 强类型系统 | 编译时发现问题 |
| API 演进 | 新增字段不影响旧客户端 |
| 自动文档 | Schema 即文档 |

## 五、注意事项

1. **GraphQL 不是 REST 的替代品**，而是补充
2. **简单 CRUD 场景 REST 更合适**
3. **学习曲线相对较高**
4. **缓存策略比 REST 复杂**
5. **适合复杂数据关系和多端场景**
