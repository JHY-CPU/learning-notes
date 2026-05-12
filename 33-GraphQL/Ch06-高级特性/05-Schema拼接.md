# Schema Stitching (已废弃，推荐 Federation)

## 一、Schema 拼接方式

```typescript
// Apollo Federation 是推荐方案
// Schema Stitching 已不再推荐
// 以下为了解原理

import { stitchSchemas } from '@graphql-tools/stitch';

const stitchedSchema = stitchSchemas({
  subschemas: [
    { schema: userServiceSchema },
    { schema: orderServiceSchema },
    { schema: productServiceSchema },
  ],
  mergeTypes: true,
});
```

## 二、Federation vs Stitching

```yaml
对比:
  Federation:
    - Apollo 官方推荐
    - 子图各自声明扩展
    - Gateway 统一编排
    - 实体自动解析
    - 生态活跃

  Stitching (旧):
    - 手动合并 Schema
    - 需要定义所有类型映射
    - 配置复杂
    - 已不推荐使用
```

## 三、迁移建议

```yaml
从 Stitching 迁移到 Federation:
  Step 1: 梳理现有类型
    - 识别核心实体
    - 确定子图边界

  Step 2: 拆分子图
    - 每个服务一个子图
    - 定义 @key 指令

  Step 3: 配置 Gateway
    - 注册子图
    - 验证编译

  Step 4: 灰度切换
    - 新旧并行
    - 验证功能
    - 全量切换
```

## 四、Apollo Federation 2 示例

```graphql
# 用户子图 (user-subgraph)
extend schema @link(url: "https://specs.apollo.dev/federation/v2.0", import: ["@key", "@shareable"])

type User @key(fields: "id") {
  id: ID!
  name: String!
  email: String!
}

type Query {
  user(id: ID!): User
  me: User
}
```

```graphql
# 订单子图 (order-subgraph)
extend schema @link(url: "https://specs.apollo.dev/federation/v2.0", import: ["@key", "@shareable", "@external"])

type User @key(fields: "id") {
  id: ID! @external
  orders: [Order!]!
}

type Order @key(fields: "id") {
  id: ID!
  total: Float!
  status: String!
  user: User!
}

type Query {
  order(id: ID!): Order
}
```

```graphql
# 商品子图 (product-subgraph)
type Product @key(fields: "id") {
  id: ID!
  name: String!
  price: Float!
}

type Order @key(fields: "id") {
  id: ID! @external
  items: [OrderItem!]!
}

type OrderItem {
  product: Product!
  quantity: Int!
}
```

## 五、Gateway 配置

```typescript
// Apollo Gateway 配置
import { ApolloServer } from '@apollo/server';
import { ApolloGateway, IntrospectAndCompose } from '@apollo/gateway';
import { startStandaloneServer } from '@apollo/server/standalone';

const gateway = new ApolloGateway({
  supergraphSdl: new IntrospectAndCompose({
    subgraphs: [
      { name: 'users', url: 'http://localhost:4001/graphql' },
      { name: 'orders', url: 'http://localhost:4002/graphql' },
      { name: 'products', url: 'http://localhost:4003/graphql' },
    ],
  }),
  buildService({ url }) {
    return new RemoteGraphQLDataSource({
      url,
      willSendRequest({ request, context }) {
        // 传递认证信息到子图
        request.http?.headers.set('authorization', context.authToken);
      },
    });
  },
});

const server = new ApolloServer({ gateway, subscriptions: false });
startStandaloneServer(server, { listen: { port: 4000 } });
```

## 六、注意事项

1. **Federation 是微服务 GraphQL 的标准方案**
2. **Stitching 已不推荐新项目使用**
3. **迁移要渐进式**
4. **实体关系要仔细设计**
5. **Gateway 要做高可用**
6. **@key 指令定义实体的主键**
7. **子图之间通过实体引用关联**，避免直接调用
