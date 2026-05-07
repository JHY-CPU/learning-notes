# Federation 联邦

## 一、架构概述

```
Apollo Federation:
┌──────────────────────────┐
│     Apollo Gateway       │ ← 统一入口
│     (Router/Gateway)     │
└──┬──────┬──────┬─────────┘
   │      │      │
   ▼      ▼      ▼
┌──────┐┌──────┐┌──────┐
│ 用户  ││ 订单  ││ 商品  │ ← 各自维护 Schema
│ 服务  ││ 服务  ││ 服务  │
└──────┘└──────┘└──────┘
```

## 二、子图定义

```graphql
# 用户服务 (users subgraph)
extend schema @link(url: "https://specs.apollo.dev/federation/v2.0", import: ["@key", "@shareable"])

type User @key(fields: "id") {
  id: ID!
  name: String!
  email: String!
}

type Query {
  user(id: ID!): User
  users: [User!]!
}
```

```graphql
# 订单服务 (orders subgraph)
extend schema @link(url: "https://specs.apollo.dev/federation/v2.0", import: ["@key", "@shareable"])

type Order @key(fields: "id") {
  id: ID!
  total: Float!
  status: String!
  user: User!
}

# 扩展用户类型 - 添加 orders 字段
extend type User @key(fields: "id") {
  id: ID! @external
  orders: [Order!]!
}

type Query {
  order(id: ID!): Order
  orders: [Order!]!
}
```

## 三、网关配置

```yaml
# router.yaml
supergraph:
  listen: 0.0.0.0:4000

subgraphs:
  users:
    routing_url: http://users-service:4001/graphql
    schema:
      subgraph_url: http://users-service:4001/graphql
  orders:
    routing_url: http://orders-service:4002/graphql
    schema:
      subgraph_url: http://orders-service:4002/graphql
  products:
    routing_url: http://products-service:4003/graphql
    schema:
      subgraph_url: http://products-service:4003/graphql
```

## 四、指令详解

```graphql
# @key - 实体主键
type User @key(fields: "id") { }
type Product @key(fields: "sku") { }

# @external - 外部字段
extend type User @key(fields: "id") {
  id: ID! @external      # 声明来自其他子图
  orders: [Order!]!      # 本子图提供
}

# @shareable - 共享字段
type Product @key(fields: "id") @shareable {
  id: ID!
  name: String! @shareable
}

# @override - 覆盖字段
type Product @key(fields: "id") {
  id: ID!
  price: Float! @override(from: "legacy-products")
}
```

## 五、注意事项

1. **Federation 2 是推荐版本**
2. **实体需要 resolveReference 实现**
3. **Schema 变更要通过 Composition 检查**
4. **网关是单点，要做高可用**
5. **各子图独立部署和扩展**
