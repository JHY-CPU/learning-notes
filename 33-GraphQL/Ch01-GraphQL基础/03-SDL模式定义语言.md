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

## 五、注意事项

1. **SDL 是 Schema 的声明式写法**
2. **输入类型不能包含对象类型字段**
3. **联合类型的所有成员必须是对象类型**
4. **接口可以被多个类型实现**
5. **枚举值默认全大写命名**
