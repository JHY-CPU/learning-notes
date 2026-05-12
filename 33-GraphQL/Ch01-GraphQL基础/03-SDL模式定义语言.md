# SDL 模式定义语言

## 一、标量类型

```graphql
# 内置标量
scalar Int       # 32位整数
scalar Float     # 浮点数
scalar String    # UTF-8 字符串
scalar Boolean   # true/false
scalar ID        # 唯一标识符

# 自定义标量
scalar DateTime
scalar JSON
scalar URL
```

## 二、对象类型与接口

```graphql
# 对象类型
type Product {
  id: ID!
  name: String!
  price: Float!
  category: Category!
  reviews: [Review!]!
}

# 接口 - 多态
interface Node {
  id: ID!
}

type User implements Node {
  id: ID!
  name: String!
}

type Product implements Node {
  id: ID!
  name: String!
  price: Float!
}

# 联合类型
union SearchResult = User | Product | Order

type Query {
  search(keyword: String!): [SearchResult!]!
}
```

## 三、输入类型与枚举

```graphql
# 输入类型 - Mutation 参数
input CreateUserInput {
  name: String!
  email: String!
  password: String!
  role: UserRole = CUSTOMER
}

input UpdateUserInput {
  name: String
  email: String
}

# 枚举
enum UserRole {
  ADMIN
  CUSTOMER
  VENDOR
}

# 分页输入
input PageInput {
  first: Int = 10
  after: String
}
```

## 四、指令

```graphql
# 内置指令
directive @deprecated(reason: String = "No longer supported") on FIELD_DEFINITION | ENUM_VALUE

type Product {
  id: ID!
  name: String!
  price: Float!
  oldPrice: Float @deprecated(reason: "Use price field")
}

# 自定义指令
directive @auth(requires: Role!) on FIELD_DEFINITION

type Query {
  users: [User!]! @auth(requires: ADMIN)
  me: User! @auth(requires: CUSTOMER)
}
```

## 五、完整的 Schema 示例

```graphql
# schema.graphql - 电商系统示例

# 自定义标量
scalar DateTime
scalar Decimal

# 枚举
enum OrderStatus {
  PENDING
  CONFIRMED
  SHIPPED
  DELIVERED
  CANCELLED
}

enum SortDirection {
  ASC
  DESC
}

# 接口
interface Node {
  id: ID!
}

interface Timestamped {
  createdAt: DateTime!
  updatedAt: DateTime!
}

# 对象类型
type User implements Node & Timestamped {
  id: ID!
  name: String!
  email: String!
  orders(first: Int, after: String): OrderConnection!
  createdAt: DateTime!
  updatedAt: DateTime!
}

type Product implements Node {
  id: ID!
  name: String!
  description: String
  price: Decimal!
  category: Category!
  inStock: Boolean!
}

type Order implements Node {
  id: ID!
  orderNo: String!
  status: OrderStatus!
  items: [OrderItem!]!
  total: Decimal!
  user: User!
  createdAt: DateTime!
}

type OrderItem {
  product: Product!
  quantity: Int!
  unitPrice: Decimal!
}

type Category implements Node {
  id: ID!
  name: String!
  products(first: Int): ProductConnection!
}

# 联合类型
union SearchResult = User | Product | Order

# Connection 分页
type ProductConnection {
  edges: [ProductEdge!]!
  pageInfo: PageInfo!
  totalCount: Int!
}

type ProductEdge {
  node: Product!
  cursor: String!
}

type PageInfo {
  hasNextPage: Boolean!
  hasPreviousPage: Boolean!
  startCursor: String
  endCursor: String
}

# 输入类型
input CreateOrderInput {
  items: [OrderItemInput!]!
  addressId: ID!
  paymentMethod: PaymentMethod!
}

input OrderItemInput {
  productId: ID!
  quantity: Int!
}

enum PaymentMethod {
  ALIPAY
  WECHAT_PAY
  CREDIT_CARD
}

# 根类型
type Query {
  node(id: ID!): Node
  user(id: ID!): User
  products(
    first: Int
    after: String
    categoryId: ID
    search: String
  ): ProductConnection!
  search(keyword: String!): [SearchResult!]!
}

type Mutation {
  createOrder(input: CreateOrderInput!): Order!
  cancelOrder(id: ID!): Order!
}

type Subscription {
  orderStatusChanged(orderId: ID!): Order!
}
```

## 六、SDL 导入与合并

```typescript
// 多模块 Schema 合并
import { loadFilesSync } from '@graphql-tools/load-files';
import { mergeTypeDefs } from '@graphql-tools/merge';

const typesArray = loadFilesSync('./src/modules/**/*.graphql');
const mergedSchema = mergeTypeDefs(typesArray);
```

## 七、注意事项

1. **SDL 是 Schema 的声明式写法**
2. **输入类型不能包含对象类型字段**
3. **联合类型的所有成员必须是对象类型**
4. **接口可以被多个类型实现**
5. **枚举值默认全大写命名**
6. **推荐一个领域模块一个 .graphql 文件**
7. **使用 graphql-codegen 自动生成 TypeScript 类型**
