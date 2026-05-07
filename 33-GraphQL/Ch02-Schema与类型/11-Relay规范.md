# Relay 规范

## 一、Node 接口

```graphql
# Node 接口 - 全局唯一标识
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
}

# 全局 ID 解析
type Query {
  node(id: ID!): Node
  nodes(ids: [ID!]!): [Node]!
}
```

```java
// 全局 ID 编解码
public class GlobalIdUtil {
    public static String toGlobalId(String typeName, String id) {
        return Base64.getEncoder()
            .encodeToString((typeName + ":" + id).getBytes());
    }

    public static String[] fromGlobalId(String globalId) {
        String decoded = new String(
            Base64.getDecoder().decode(globalId));
        return decoded.split(":", 2);
    }
}
```

## 二、Connection 分页

```graphql
# Connection 模式
type UserConnection {
  edges: [UserEdge!]!
  pageInfo: PageInfo!
  totalCount: Int!
}

type UserEdge {
  node: User!
  cursor: String!
}

type PageInfo {
  hasNextPage: Boolean!
  hasPreviousPage: Boolean!
  startCursor: String
  endCursor: String
}

type Query {
  users(
    first: Int
    after: String
    last: Int
    before: String
  ): UserConnection!
}
```

## 三、Mutations 返回模式

```graphql
# Relay Mutation 模式 - 返回 payload 和修改的对象
input CreateUserInput {
  name: String!
  email: String!
}

type CreateUserPayload {
  user: User
  userEdge: UserEdge        # 用于 Connection 更新
  errors: [UserError!]!
}

type UserError {
  field: [String!]
  message: String!
}

type Mutation {
  createUser(input: CreateUserInput!): CreateUserPayload!
}
```

## 四、注意事项

1. **Relay 是 Facebook 的 GraphQL 客户端规范**
2. **Node 接口提供全局唯一 ID**
3. **Connection 是标准分页模式**
4. **全局 ID 编码类型信息**
5. **遵循 Relay 规范有利于客户端缓存**
