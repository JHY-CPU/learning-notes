# Schema 设计原则

## 一、设计流程

```
Schema 设计:
├── 1. 梳理业务领域
├── 2. 识别核心实体
├── 3. 定义实体关系
├── 4. 确定 Query/Mutation
├── 5. 设计 Input/Output
└── 6. 演进与评审
```

## 二、命名规范

```graphql
# ✓ 推荐命名
type User { }                  # PascalCase 类型
enum OrderStatus { }           # PascalCase 枚举
input CreateUserInput { }      # PascalCase 输入

type Query {
  user(id: ID!): User          # camelCase 查询
  users: [User!]!
}

type Mutation {
  createUser(input: CreateUserInput!): User!  # camelCase 变更
}

# ✓ 字段命名
type User {
  firstName: String!           # camelCase 字段
  lastName: String!
  createdAt: DateTime!         # 时间后缀
  isActive: Boolean!           # 布尔 is/has 前缀
  orderCount: Int!             # 计数 xxxCount
}

# ✗ 避免
type User {
  user_name: String            # 下划线
  UserName: String             # 大写
  orders_list: [Order]         # 后缀冗余
}
```

## 三、Schema First vs Code First

```yaml
Schema First (SDL):
  流程:
    - 先写 .graphql 文件
    - 代码生成工具生成骨架
    - 实现 Resolver
  优点: 与前端共享、文档优先
  工具: graphql-code-generator

Code First:
  流程:
    - 用代码定义类型
    - 框架自动生成 SDL
  优点: 类型安全、重构方便
  工具: graphql-java, TypeGraphQL
```

## 四、命名规范扩展

```graphql
# Mutation 命名模式
type Mutation {
  # ✓ create + 单数实体
  createUser(input: CreateUserInput!): User!
  createOrder(input: CreateOrderInput!): Order!

  # ✓ update + 单数实体
  updateUser(id: ID!, input: UpdateUserInput!): User!

  # ✓ delete/remove + 单数实体
  deleteUser(id: ID!): DeleteResult!
  removeOrderItem(orderId: ID!, itemId: ID!): Order!

  # ✓ 特定业务动作
  placeOrder(input: PlaceOrderInput!): Order!
  cancelOrder(id: ID!): Order!
  payOrder(id: ID!, method: PaymentMethod!): PaymentResult!

  # ✗ 避免
  doSomething(input: SomeInput!): Result   # 动词不明确
  userCreate(input: UserInput!): User!      # 名词在前
}
```

## 五、反向关联设计

```graphql
# ✓ 双向关联 - 便于客户端查询
type User {
  id: ID!
  name: String!
  orders(first: Int, after: String): OrderConnection!  # 用户 → 订单
}

type Order {
  id: ID!
  total: Float!
  user: User!  # 订单 → 用户
}

# ✗ 单向关联 - 客户端无法从订单查用户
type Order {
  id: ID!
  total: Float!
  userId: ID!  # 只有 ID，客户端要额外查询
}
```

## 六、分页设计

```graphql
# ✓ Connection 模式 (Relay 规范)
type Query {
  users(first: Int, after: String): UserConnection!
}

type UserConnection {
  edges: [UserEdge!]!
  pageInfo: PageInfo!
  totalCount: Int!
}

type UserEdge {
  node: User!
  cursor: String!
}

# ✗ 简单数组 - 无分页信息
type Query {
  users: [User!]!  # 一次性返回所有数据
}
```

## 七、字段可见性控制

```graphql
# 通过 Resolver 控制字段可见性，而非 Schema
type User {
  id: ID!
  name: String!
  email: String!        # 在 Resolver 中做权限判断
  salary: Float         # 在 Resolver 中做权限判断
}

# Resolver 中的权限控制
const resolvers = {
  User: {
    email: (parent, _, context) => {
      // 只有自己或管理员能看
      if (context.currentUser.id === parent.id || context.currentUser.isAdmin) {
        return parent.email;
      }
      return null;
    },
    salary: (parent, _, context) => {
      // 只有管理员能看
      if (!context.currentUser.isAdmin) return null;
      return parent.salary;
    },
  },
};
```

## 八、注意事项

1. **Schema 是 API 契约**，要与前端团队共同设计
2. **遵循 Relay 规范**（Connection/Edge/PageInfo）
3. **一个实体只有一个 owner 类型**
4. **避免深层嵌套**，建议不超过 3 层
5. **定期评审 Schema**，保持整洁
6. **用 @deprecated 替代删除字段**
7. **字段级权限在 Resolver 实现**，不要在 Schema 中定义多个版本
