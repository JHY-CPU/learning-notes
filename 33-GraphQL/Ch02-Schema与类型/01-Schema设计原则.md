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

## 四、注意事项

1. **Schema 是 API 契约**，要与前端团队共同设计
2. **遵循 Relay 规范**（Connection/Edge/PageInfo）
3. **一个实体只有一个 owner 类型**
4. **避免深层嵌套**，建议不超过 3 层
5. **定期评审 Schema**，保持整洁
