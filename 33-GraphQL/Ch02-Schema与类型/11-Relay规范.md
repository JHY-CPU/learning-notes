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

## 四、Connection 实现 (Node.js)

```javascript
// Relay Connection 实现
const resolvers = {
  Query: {
    users: async (_, args) => {
      const { first, after, last, before } = args;
      const limit = first || last || 10;

      let cursor = null;
      let direction = 'forward';

      if (after) {
        cursor = decodeCursor(after);
        direction = 'forward';
      } else if (before) {
        cursor = decodeCursor(before);
        direction = 'backward';
      }

      const { items, hasNextPage, hasPreviousPage } = await db.users.findMany({
        cursor,
        limit: limit + 1, // 多取 1 条判断是否有下页
        direction,
      });

      const hasMore = items.length > limit;
      const nodes = hasMore ? items.slice(0, limit) : items;

      return {
        edges: nodes.map((node) => ({
          node,
          cursor: encodeCursor(node.id),
        })),
        pageInfo: {
          hasNextPage: direction === 'forward' ? hasMore : false,
          hasPreviousPage: direction === 'backward' ? hasMore : !!cursor,
          startCursor: nodes.length > 0 ? encodeCursor(nodes[0].id) : null,
          endCursor: nodes.length > 0 ? encodeCursor(nodes[nodes.length - 1].id) : null,
        },
        totalCount: await db.users.count(),
      };
    },
  },
};
```

## 五、Connection TypeScript 类型

```typescript
// 通用 Connection 类型
interface Connection<T> {
  edges: Edge<T>[];
  pageInfo: PageInfo;
  totalCount: number;
}

interface Edge<T> {
  node: T;
  cursor: string;
}

interface PageInfo {
  hasNextPage: boolean;
  hasPreviousPage: boolean;
  startCursor: string | null;
  endCursor: string | null;
}

// 使用
type UserConnection = Connection<User>;
type ProductConnection = Connection<Product>;
```

## 六、客户端使用 Connection

```javascript
// Apollo Client + Relay Connection
import { useQuery, gql } from '@apollo/client';

const GET_USERS = gql`
  query GetUsers($first: Int!, $after: String) {
    users(first: $first, after: $after) {
      edges {
        node {
          id
          name
          email
        }
        cursor
      }
      pageInfo {
        hasNextPage
        endCursor
      }
    }
  }
`;

function UserList() {
  const { data, fetchMore } = useQuery(GET_USERS, {
    variables: { first: 10 },
  });

  const loadMore = () => {
    if (data?.users.pageInfo.hasNextPage) {
      fetchMore({
        variables: {
          after: data.users.pageInfo.endCursor,
        },
        // Apollo 合并策略自动合并 edges
      });
    }
  };

  return (
    <div>
      {data?.users.edges.map(({ node }) => (
        <UserCard key={node.id} user={node} />
      ))}
      {data?.users.pageInfo.hasNextPage && (
        <button onClick={loadMore}>加载更多</button>
      )}
    </div>
  );
}
```

## 七、注意事项

1. **Relay 是 Facebook 的 GraphQL 客户端规范**
2. **Node 接口提供全局唯一 ID**
3. **Connection 是标准分页模式**
4. **全局 ID 编码类型信息**
5. **遵循 Relay 规范有利于客户端缓存**
6. **cursor 用 Base64 编码**，避免暴露内部 ID
7. **Apollo Cache 需要配置 typePolicies 合并 Connection**
