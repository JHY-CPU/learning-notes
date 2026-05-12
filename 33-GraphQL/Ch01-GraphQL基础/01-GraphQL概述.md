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

## 五、GraphQL 起源与发展

GraphQL 由 Facebook 于 2012 年内部开发，用于解决移动客户端数据获取效率低下的问题。2015 年正式开源，2018 年成立 GraphQL Foundation（隶属于 Linux Foundation）推动标准化。

```yaml
发展里程碑:
  2012: Facebook 内部开发
  2015: 开源发布
  2016: Apollo 推出客户端和服务端工具
  2018: GraphQL Foundation 成立
  2021: 规范以 Open Spec 形式发布
  2023: Apollo Federation 2 成熟
  2024: Defer/Stream 提案推进
```

## 六、核心架构模式

```
典型 GraphQL 架构:

客户端 (React/Vue/iOS/Android)
        │
        ▼
┌────────────────────┐
│   GraphQL Gateway  │ ← 认证、限流、查询分析
├────────────────────┤
│   Schema Registry  │ ← 版本管理与兼容性检查
└────────┬───────────┘
         │
    ┌────┴────┬─────────┐
    ▼         ▼         ▼
 用户服务  订单服务  商品服务
 (REST/gRPC/DB) (REST/gRPC/DB) (REST/gRPC/DB)
```

## 七、Resolver 实现示例

```typescript
// TypeScript + Apollo Server 实现
import { ApolloServer, gql } from '@apollo/server';

const typeDefs = gql`
  type Query {
    user(id: ID!): User
  }
  type User {
    id: ID!
    name: String!
    email: String!
  }
`;

const resolvers = {
  Query: {
    user: async (_, { id }, context) => {
      // context 包含认证信息、DataLoader 等
      return context.dataSources.userAPI.getUser(id);
    },
  },
  User: {
    // 字段级 Resolver - 按需加载
    email: async (parent, _, context) => {
      if (!context.currentUser.isAdmin) {
        return null; // 权限控制
      }
      return parent.email;
    },
  },
};

const server = new ApolloServer({ typeDefs, resolvers });
```

## 八、与其他 API 方案对比

| 方案 | 数据获取 | 类型安全 | 实时 | 学习成本 | 适用规模 |
|------|----------|----------|------|----------|----------|
| REST | 固定 | 弱 | 需额外实现 | 低 | 小-中 |
| GraphQL | 灵活 | 强 | Subscription | 中 | 中-大 |
| gRPC | 固定 | 强 | Stream | 高 | 内部服务 |
| tRPC | 灵活 | 极强 | Subscription | 低 | TS 全栈 |

## 九、注意事项

1. **GraphQL 不是 REST 的替代品**，而是补充
2. **简单 CRUD 场景 REST 更合适**
3. **学习曲线相对较高**
4. **缓存策略比 REST 复杂**
5. **适合复杂数据关系和多端场景**
6. **优先考虑 Apollo 生态**，工具链最完善
7. **Schema 设计是最重要的环节**，需前后端共同参与
