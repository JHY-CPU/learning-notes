# Defer 与 Stream

## 一、@defer - 延迟字段

```graphql
# 延迟加载非关键字段
query GetUser($id: ID!) {
  user(id: $id) {
    name
    email           # 立即返回
    ... @defer {
      orders {      # 延迟返回
        id
        total
        status
      }
    }
    ... @defer {
      reviews {     # 延迟返回
        rating
        comment
      }
    }
  }
}

# 响应流:
# 第一个响应 - 快速数据
{
  "data": { "user": { "name": "张三", "email": "zhang@test.com" } },
  "hasNext": true
}
# 第二个响应 - 延迟的 orders
{
  "incremental": [{ "data": { "orders": [...] }, "path": ["user"] }],
  "hasNext": true
}
# 第三个响应 - 延迟的 reviews
{
  "incremental": [{ "data": { "reviews": [...] }, "path": ["user"] }],
  "hasNext": false
}
```

## 二、@stream - 流式列表

```graphql
# 列表流式返回
query GetProducts {
  products {
    id
    name
    reviews @stream(initialCount: 5) {
      # 先返回5条，后续流式返回
      rating
      comment
      user { name }
    }
  }
}
```

## 三、客户端使用

```typescript
// Apollo Client @defer 支持
import { useQuery } from '@apollo/client';

const GET_USER = gql`
  query GetUser($id: ID!) {
    user(id: $id) {
      name
      email
      ... @defer {
        orders {
          id
          total
        }
      }
    }
  }
`;

function UserProfile({ userId }) {
  const { data, loading } = useQuery(GET_USER, {
    variables: { id: userId },
  });

  return (
    <div>
      {/* 立即显示 */}
      <h1>{data?.user.name}</h1>
      <p>{data?.user.email}</p>

      {/* 延迟加载 - 随数据到达自动更新 */}
      {data?.user.orders ? (
        <OrderList orders={data.user.orders} />
      ) : (
        <OrdersLoading />
      )}
    </div>
  );
}
```

## 四、注意事项

1. **@defer/@stream 还是实验性特性**
2. **服务端需要支持 Incremental Delivery 协议**
3. **客户端要支持 multipart 响应**
4. **适合慢字段延迟加载**
5. **改善首屏加载体验**
